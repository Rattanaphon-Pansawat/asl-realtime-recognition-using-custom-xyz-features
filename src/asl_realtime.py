import os, sys, math, json, time, csv, string, itertools, pathlib
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2

# ----- MediaPipe import with Raspberry Pi fallback -----
try:
    import mediapipe as mp
except Exception as e:
    try:
        import mediapipe_rpi4 as mp  # fallback on some Pi builds
    except Exception as e2:
        raise RuntimeError(
            "Cannot import mediapipe. Try: pip install mediapipe  (or mediapipe-rpi4 on Raspberry Pi)"
        )

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd


# ----------------------- Utility Math -----------------------
def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / (n + 1e-9)

def angle_between(u: np.ndarray, v: np.ndarray) -> float:
    """Return angle in degrees between vectors u and v."""
    uu, vv = unit(u), unit(v)
    dot = np.clip(np.dot(uu, vv), -1.0, 1.0)
    return float(np.degrees(np.arccos(dot)))

def palm_normal(pts: np.ndarray) -> np.ndarray:
    # Use wrist->index_MCP (5) and wrist->pinky_MCP (17)
    a = pts[5] - pts[0]
    b = pts[17] - pts[0]
    n = np.cross(a, b)
    return unit(n)

def orientation_angles_from_normal(n: np.ndarray) -> Tuple[float,float,float]:
    """Return (pitch, yaw, roll) in degrees from palm normal n (x,y,z)."""
    x,y,z = float(n[0]), float(n[1]), float(n[2])
    pitch = math.degrees(math.atan2(x, math.sqrt(y*y + z*z)))
    yaw   = math.degrees(math.atan2(y, math.sqrt(x*x + z*z)))
    roll  = math.degrees(math.atan2(math.sqrt(x*x + y*y), z + 1e-9))
    return (pitch, yaw, roll)

# Mediapipe hand indices
WRIST = 0
TH_CMC, TH_MCP, TH_IP, TH_TIP = 1,2,3,4
IX_MCP, IX_PIP, IX_DIP, IX_TIP = 5,6,7,8
MD_MCP, MD_PIP, MD_DIP, MD_TIP = 9,10,11,12
RG_MCP, RG_PIP, RG_DIP, RG_TIP = 13,14,15,16
PK_MCP, PK_PIP, PK_DIP, PK_TIP = 17,18,19,20

FINGERS = {
    "thumb":   [TH_CMC, TH_MCP, TH_IP, TH_TIP],
    "index":   [IX_MCP, IX_PIP, IX_DIP, IX_TIP],
    "middle":  [MD_MCP, MD_PIP, MD_DIP, MD_TIP],
    "ring":    [RG_MCP, RG_PIP, RG_DIP, RG_TIP],
    "pinky":   [PK_MCP, PK_PIP, PK_DIP, PK_TIP],
}
TIP_IDS = [TH_TIP, IX_TIP, MD_TIP, RG_TIP, PK_TIP]

# ----------------------- Feature Extractor -----------------------
@dataclass
class HandFeatures:
    # Core vector to feed ML
    vector: np.ndarray
    # Extra parsed values useful for rules/overlay
    flex_angles: Dict[str, Dict[str, float]]  # finger -> {"pip":deg,"dip":deg}
    orientation: Dict[str, float]            # pitch,yaw,roll
    tip_dists: Dict[Tuple[int,int], float]   # pairwise tip distances (normalized)
    meta: Dict[str, float]                   # scale, etc.

class FeatureExtractor:
    def __init__(self):
        # Which features and order (deterministic)
        self.feature_names = []
        # build a stable schema once
        self._build_schema()

    def _build_schema(self):
        names = []

        # 1) normalized coords diffs (tip - wrist) for all 21 points -> 63
        for i in range(21):
            for ax in ["dx","dy","dz"]:
                names.append(f"lm{i}_{ax}")

        # 2) flexion angles at PIP and DIP for 4 non-thumb fingers + thumb-IP angle -> 9
        for f in ["index","middle","ring","pinky"]:
            names += [f"{f}_pip_deg", f"{f}_dip_deg"]
        names += ["thumb_ip_deg"]

        # 3) palm orientation angles -> 3
        names += ["palm_pitch","palm_yaw","palm_roll"]

        # 4) pairwise tip distances (C(5,2)=10)
        for (a,b) in itertools.combinations(TIP_IDS, 2):
            names.append(f"tipdist_{a}_{b}")

        # 5) selected structural distances (tip->wrist) -> 5
        for t in TIP_IDS:
            names.append(f"tip2wrist_{t}")

        self.feature_names = names

    def get_schema(self) -> List[str]:
        return list(self.feature_names)

    def _finger_flex_angles(self, pts: np.ndarray) -> Dict[str, Dict[str,float]]:
        out = {}
        # For 4 fingers
        for f in ["index","middle","ring","pinky"]:
            mcp,pip,dip,tip = FINGERS[f]
            v1 = pts[mcp] - pts[pip]
            v2 = pts[dip] - pts[pip]
            pip_deg = angle_between(v1, v2)

            v3 = pts[pip] - pts[dip]
            v4 = pts[tip] - pts[dip]
            dip_deg = angle_between(v3, v4)

            out[f] = {"pip": pip_deg, "dip": dip_deg}
        # thumb
        v1 = pts[TH_MCP] - pts[TH_IP]
        v2 = pts[TH_TIP] - pts[TH_IP]
        thumb_ip = angle_between(v1, v2)
        out["thumb"] = {"ip": thumb_ip}
        return out

    def _extended_state(self, flex: Dict[str,Dict[str,float]]) -> Dict[str,str]:
        """Return 'extended' / 'half' / 'curled' per finger using angles."""
        state = {}
        for f in ["index","middle","ring","pinky"]:
            pip = flex[f]["pip"]
            dip = flex[f]["dip"]
            if pip > 160 and dip > 160:
                state[f] = "extended"
            elif pip < 105 and dip < 105:
                state[f] = "curled"
            else:
                state[f] = "half"
        # thumb heuristic
        tip = flex["thumb"]["ip"]
        state["thumb"] = "extended" if tip > 150 else ("curled" if tip < 110 else "half")
        return state

    def extract(self, landmarks: List) -> HandFeatures:
        """landmarks: mediapipe list of 21 landmarks (normalized image coords)."""
        pts = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)

        # normalize by wrist-centered scale for invariance
        wrist = pts[WRIST].copy()
        pts_rel = pts - wrist

        # scale: mean distance wrist->MCP of index/middle/ring/pinky
        mcp_ids = [IX_MCP, MD_MCP, RG_MCP, PK_MCP]
        scale = float(np.mean([np.linalg.norm(pts_rel[i]) for i in mcp_ids]) + 1e-9)
        pts_n = pts_rel / scale

        # 1) coord diffs
        feats = []
        for i in range(21):
            feats.extend(list(pts_n[i]))  # dx,dy,dz

        # 2) finger flexion
        flex = self._finger_flex_angles(pts_n)
        for f in ["index","middle","ring","pinky"]:
            feats.extend([flex[f]["pip"], flex[f]["dip"]])
        feats.append(flex["thumb"]["ip"])

        # 3) palm orientation (from normal)
        n = palm_normal(pts_n)
        pitch, yaw, roll = orientation_angles_from_normal(n)
        feats.extend([pitch, yaw, roll])

        # 4) pairwise tip distances
        tip_d = {}
        for (a,b) in itertools.combinations(TIP_IDS, 2):
            d = float(np.linalg.norm(pts_n[a] - pts_n[b]))
            feats.append(d)
            tip_d[(a,b)] = d

        # 5) tip->wrist distances
        for t in TIP_IDS:
            feats.append(float(np.linalg.norm(pts_n[t])))

        vec = np.array(feats, dtype=np.float32)

        extra = {
            "flex_angles": flex,
            "orientation": {"pitch":pitch,"yaw":yaw,"roll":roll},
            "tip_dists": tip_d,
            "meta": {"scale": scale}
        }
        return HandFeatures(vector=vec, flex_angles=flex, orientation=extra["orientation"],
                            tip_dists=tip_d, meta=extra["meta"])

# ----------------------- Heuristic Classifier -----------------------
class HeuristicASL:
    """Rule-based scoring; returns best letter and score in [0,1]."""
    letters = [c for c in string.ascii_uppercase if c not in ["J","Z"]]

    def __init__(self):
        pass

    def _states(self, feats: HandFeatures) -> Dict[str,str]:
        return FeatureExtractor()._extended_state(feats.flex_angles)

    def _close(self, feats: HandFeatures, a:int, b:int, thresh:float=0.23) -> bool:
        return feats.tip_dists.get((min(a,b), max(a,b)), 9.9) < thresh

    def _apart(self, feats: HandFeatures, a:int, b:int, thresh:float=0.35) -> bool:
        return feats.tip_dists.get((min(a,b), max(a,b)), 0.0) > thresh

    def score(self, feats: HandFeatures) -> Tuple[str, float, Dict[str,float]]:
        s = self._states(feats)
        pitch, yaw, roll = feats.orientation["pitch"], feats.orientation["yaw"], feats.orientation["roll"]

        # handy shorthands
        thumb, index, middle, ring, pinky = "thumb","index","middle","ring","pinky"

        sc = {}

        # A: all curled, thumb alongside index
        cond = (s[index]==s[middle]==s[ring]==s[pinky]=="curled")
        near_index_mcp = feats.vector[ (IX_MCP*3) ]  # dx of MCP to wrist (proxy for side); not robust; use distance tip->IX_MCP
        d_ti = np.linalg.norm(feats.vector[TH_TIP*3:TH_TIP*3+3] - feats.vector[IX_MCP*3:IX_MCP*3+3])
        sc["A"] = 0.9 if cond and d_ti<0.35 and s[thumb]!="curled" else 0.0

        # B: 4 fingers extended together; thumb across palm (not extended)
        together = (not self._apart(feats, IX_TIP, MD_TIP, 0.28) and
                    not self._apart(feats, MD_TIP, RG_TIP, 0.28) and
                    not self._apart(feats, RG_TIP, PK_TIP, 0.30))
        sc["B"] = 0.9 if (s[index]==s[middle]==s[ring]==s[pinky]=="extended" and together and s[thumb]!="extended") else 0.0

        # C: all half-bent; thumb-index apart but curved
        halfs = (s[index]=="half" and s[middle]=="half" and s[ring]=="half" and s[pinky]=="half")
        sc["C"] = 0.8 if halfs and self._apart(feats, TH_TIP, IX_TIP, 0.28) else 0.0

        # D: index extended; others curled; thumb close to index tip (circle)
        sc["D"] = 0.9 if (s[index]=="extended" and s[middle]=="curled" and s[ring]=="curled" and s[pinky]=="curled" and self._close(feats, TH_TIP, IX_TIP, 0.18)) else 0.0

        # E: all curled; thumb in front touching fingers (thumb close to multiple tips)
        near_ix = self._close(feats, TH_TIP, IX_TIP, 0.20)
        near_md = self._close(feats, TH_TIP, MD_TIP, 0.22)
        sc["E"] = 0.8 if (s[index]==s[middle]==s[ring]==s[pinky]=="curled" and near_ix and near_md) else 0.0

        # F: thumb-index circle; other 3 extended
        sc["F"] = 0.9 if self._close(feats, TH_TIP, IX_TIP, 0.18) and s[middle]==s[ring]==s[pinky]=="extended" else 0.0

        # G: thumb+index extended parallel; others curled
        sc["G"] = 0.8 if (s[thumb]=="extended" and s[index]=="extended" and s[middle]==s[ring]==s[pinky]=="curled") else 0.0

        # H: index+middle extended together; others curled
        sc["H"] = 0.85 if (s[index]==s[middle]=="extended" and not self._apart(feats, IX_TIP, MD_TIP, 0.30) and s[ring]==s[pinky]=="curled") else 0.0

        # I: pinky extended only
        sc["I"] = 0.95 if (s[pinky]=="extended" and s[index]==s[middle]==s[ring]=="curled") else 0.0

        # K: index+middle extended with small spread; thumb touching middle
        sc["K"] = 0.8 if (s[index]==s[middle]=="extended" and self._apart(feats, IX_TIP, MD_TIP, 0.25) and self._close(feats, TH_TIP, MD_TIP, 0.22) and s[ring]==s[pinky]=="curled") else 0.0

        # L: index+thumb like 90°, others curled
        sc["L"] = 0.9 if (s[index]=="extended" and s[thumb]=="extended" and s[middle]==s[ring]==s[pinky]=="curled" and self._apart(feats, TH_TIP, IX_TIP, 0.30)) else 0.0

        # M: 3 fingers over thumb (กำปั้นมีนิ้วโป้งซ่อน)
        sc["M"] = 0.6 if (s[index]==s[middle]==s[ring]=="curled" and s[pinky] in ["curled","half"] and s[thumb]=="curled") else 0.0

        # N: 2 fingers over thumb
        sc["N"] = 0.6 if (s[index]==s[middle]=="curled" and s[ring] in ["curled","half"] and s[pinky] in ["curled","half"] and s[thumb]=="curled") else 0.0

        # O: ทุกนิ้วโค้งเข้าหากัน โป้งแตะใกล้นิ้วชี้
        sc["O"] = 0.85 if (s[index]==s[middle]==s[ring]==s[pinky]=="half" and self._close(feats, TH_TIP, IX_TIP, 0.20)) else 0.0

        # P: คล้าย K แต่ฝ่ามือลง (pitch/roll)
        sc["P"] = 0.75 if (sc.get("K",0)>0 and pitch < -10) else 0.0

        # Q: คล้าย G แต่ฝ่ามือลง
        sc["Q"] = 0.75 if (sc.get("G",0)>0 and pitch < -10) else 0.0

        # R: ชี้+กลาง ไขว้ (ยากด้วย landmark 2D) — ใช้ใกล้ปลาย แต่ห่างที่ MCP
        d_tip = feats.tip_dists[(min(IX_TIP,MD_TIP),max(IX_TIP,MD_TIP))]
        base_gap = np.linalg.norm(feats.vector[IX_MCP*3:IX_MCP*3+3] - feats.vector[MD_MCP*3:MD_MCP*3+3])
        sc["R"] = 0.6 if (s[index]==s[middle]=="extended" and d_tip<0.20 and base_gap>0.25 and s[ring]==s[pinky]=="curled") else 0.0

        # S: กำปั้นนิ้วโป้งพาดหน้า
        sc["S"] = 0.8 if (s[index]==s[middle]==s[ring]==s[pinky]=="curled" and s[thumb]!="extended" and not sc.get("E",0)) else 0.0

        # T: โป้งสอดระหว่างนิ้วชี้-กลาง
        sc["T"] = 0.7 if (s[index]!="extended" and s[middle]!="extended" and self._close(feats, TH_TIP, IX_PIP, 0.22)) else 0.0

        # U: ชี้+กลาง ชิดกันชูขึ้น; ที่เหลืองอ
        sc["U"] = 0.85 if (s[index]==s[middle]=="extended" and not self._apart(feats, IX_TIP, MD_TIP, 0.23) and s[ring]==s[pinky]=="curled") else 0.0

        # V: ชี้+กลาง กางเป็น V
        sc["V"] = 0.9 if (s[index]==s[middle]=="extended" and self._apart(feats, IX_TIP, MD_TIP, 0.28) and s[ring]==s[pinky]=="curled") else 0.0

        # W: ชี้+กลาง+นาง ยกขึ้น
        sc["W"] = 0.9 if (s[index]==s[middle]==s[ring]=="extended" and s[pinky]!="extended" and s[thumb]!="extended") else 0.0

        # X: นิ้วชี้งอเป็นตะขอ; อื่นกำ
        sc["X"] = 0.85 if (s[index]=="half" and s[middle]==s[ring]==s[pinky]=="curled" and s[thumb]!="extended") else 0.0

        # Y: โป้ง+ก้อย ชู
        sc["Y"] = 0.95 if (s[thumb]=="extended" and s[pinky]=="extended" and s[index]==s[middle]==s[ring]=="curled") else 0.0

        # choose best
        best = max(sc.items(), key=lambda kv: kv[1])
        return best[0], best[1], sc

# ----------------------- Dataset / ML -----------------------
class ASLDataset:
    def __init__(self):
        self.rows: List[Tuple[np.ndarray, str]] = []
        self.schema = FeatureExtractor().get_schema()

    def add(self, vec: np.ndarray, label: str):
        self.rows.append((vec.astype(np.float32), label))

    def to_dataframe(self) -> pd.DataFrame:
        X = np.stack([r[0] for r in self.rows], axis=0) if self.rows else np.zeros((0,len(self.schema)))
        y = [r[1] for r in self.rows]
        df = pd.DataFrame(X, columns=self.schema)
        df["label"] = y
        return df

    def save_csv(self, path="asl_features.csv"):
        df = self.to_dataframe()
        df.to_csv(path, index=False)
        print(f"[saved] {path}  ({len(df)} samples)")

    def load_csv(self, path="asl_features.csv"):
        df = pd.read_csv(path)
        self.schema = [c for c in df.columns if c!="label"]
        X = df[self.schema].values.astype(np.float32)
        y = df["label"].tolist()
        self.rows = [(X[i], y[i]) for i in range(len(y))]
        print(f"[loaded] {path}  ({len(y)} samples)")

class ASLClassifier:
    def __init__(self, kind="knn"):
        self.kind = kind
        self.model = KNeighborsClassifier(n_neighbors=5) if kind=="knn" else SVC(kernel="rbf", probability=True)
        self.labels: List[str] = []

    def fit(self, dataset: ASLDataset):
        if not dataset.rows:
            print("[fit] dataset is empty")
            return
        X = np.stack([r[0] for r in dataset.rows], axis=0)
        y = np.array([r[1] for r in dataset.rows])
        self.labels = sorted(list(set(y.tolist())))
        self.model.fit(X, y)
        print(f"[fit] trained {self.kind} on {len(y)} samples, {len(self.labels)} classes: {self.labels}")

    def predict(self, vec: np.ndarray) -> Tuple[str, float]:
        if not self.labels:
            return "?", 0.0
        proba = None
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(vec.reshape(1,-1))[0]
            idx = int(np.argmax(proba))
            return self.model.classes_[idx], float(np.max(proba))
        pred = self.model.predict(vec.reshape(1,-1))[0]
        return pred, 1.0

    def evaluate(self, dataset: ASLDataset, k=5):
        if not dataset.rows:
            print("[eval] dataset empty")
            return
        X = np.stack([r[0] for r in dataset.rows], axis=0)
        y = np.array([r[1] for r in dataset.rows])
        skf = StratifiedKFold(n_splits=min(k, len(set(y))), shuffle=True, random_state=0)
        scores = cross_val_score(self.model, X, y, cv=skf)
        print(f"[CV] {self.kind}  k={len(scores)}  acc={scores.mean():.3f} ± {scores.std():.3f}")

        # one train-test split for confusion
        self.model.fit(X, y)
        yhat = self.model.predict(X)
        cm = confusion_matrix(y, yhat, labels=sorted(list(set(y))))
        print("[Confusion Matrix] rows=true, cols=pred")
        print(pd.DataFrame(cm, index=self.model.classes_, columns=self.model.classes_))
        print(classification_report(y, yhat))

# ----------------------- MediaPipe Tracker -----------------------
class HandTracker:
    def __init__(self, max_num_hands=1):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

    def process(self, frame_rgb):
        return self.hands.process(frame_rgb)

    def draw(self, frame_bgr, landmarks):
        self.mp_draw.draw_landmarks(
            frame_bgr, landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_styles.get_default_hand_landmarks_style(),
            self.mp_styles.get_default_hand_connections_style(),
        )

# ----------------------- Integration Hook (Dobot etc.) -----------------------
def on_prediction(letter: str):
    """Put your robot/serial/network trigger here.
    Example for Dobot (pseudo):
      import serial
      ser = serial.Serial('COM4', 115200)
      if letter == 'A': ser.write(b'MOVE X ...')
    """
    pass

# ----------------------- Main App -----------------------
def main():
    print("ASL Real-time (A–Z without J,Z) | H=toggle Heuristic/ML, [ ]=change label, C=collect, T=train, E=eval, S=save, L=load, Q=quit")
    label_list = [c for c in string.ascii_uppercase if c not in ["J","Z"]]
    label_idx = 0
    current_label = label_list[label_idx]

    cap = None
    # Try default, then V4L2 (Pi)
    for cam_arg in [(0, ), (0, cv2.CAP_V4L2)]:
        try:
            cap = cv2.VideoCapture(*cam_arg)
            if cap.isOpened():
                break
        except:
            pass
    if cap is None or not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    tracker = HandTracker()
    fx = FeatureExtractor()
    heur = HeuristicASL()
    data = ASLDataset()
    clf = ASLClassifier(kind="knn")
    use_heuristic = True

    last_pred = "?"
    last_conf  = 0.0


    # ------------------- UI fix -------------------
    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = tracker.process(rgb)

        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            tracker.draw(frame, lm)
            feats = fx.extract(lm.landmark)

            if use_heuristic:
                letter, score, _ = heur.score(feats)
                last_pred, last_conf = letter, score
            else:
                letter, conf = clf.predict(feats.vector)
                last_pred, last_conf = letter, conf

            # trigger hook
            if last_conf > 0.8 and last_pred in label_list:
                on_prediction(last_pred)

            # ------------------- Use Roboto Font -------------------
            font_path = "E:/asl_xyz/src/Roboto_Condensed-Regular.ttf"  # Path to the Roboto font
            font = ImageFont.truetype(font_path, 20)  # Load the font

            # Convert frame to PIL image to use custom font
            pil_img = Image.fromarray(frame)
            draw = ImageDraw.Draw(pil_img)

            # Overlay info with Roboto font
            draw.text((10, 20), f"Pred: {last_pred}  conf={last_conf:.2f}  mode={'Heuristic' if use_heuristic else 'ML'}", font=font, 
                      fill=(0, 0, 0))

            # Convert back to OpenCV format
            frame = np.array(pil_img)


        # วาดพื้นหลังสีขาวสำหรับ Menu ที่ขอบล่างซ้าย
        cv2.rectangle(frame, (0, frame.shape[0]-60), (frame.shape[1], frame.shape[0]), (255, 255, 255), -1)

        # UI texts with Roboto font
        pil_img = Image.fromarray(frame)
        draw = ImageDraw.Draw(pil_img)

        # Draw label and buttons with Roboto font
        draw.text((10, frame.shape[0]-55), f"Label: {last_pred}", font=font, fill=(0, 0, 0))
        draw.text((10, frame.shape[0]-30), f"C Capture  H Toggle  T Train  E Eval  S Save  L Load  Q Quit", font=font, fill=(0, 0, 0))


        # Convert back to OpenCV format
        frame = np.array(pil_img)


        cv2.imshow("ASL Realtime", frame)
        k = cv2.waitKey(1) & 0xFF

        # ตรวจสอบการปิดหน้าต่าง (กากบาท)
        if cv2.getWindowProperty("ASL Realtime", cv2.WND_PROP_VISIBLE) < 1:
            break

        # ตรวจสอบการกดคีย์
        if k == ord('q') or k == ord('Q'):
            break
        elif k == ord('h') or k == ord('H'):
            use_heuristic = not use_heuristic
        elif k == ord('c') or k == ord('C'):
            if res and res.multi_hand_landmarks:
                feats = fx.extract(res.multi_hand_landmarks[0].landmark)
                data.add(feats.vector, current_label)
                print(f"[add] {current_label}  total={len(data.rows)}")
        elif k == ord('t') or k == ord('T'):
            clf.fit(data)
        elif k == ord('e') or k == ord('E'):
            clf.evaluate(data, k=5)
        elif k == ord('s') or k == ord('S'):
            data.save_csv()
        elif k == ord('l') or k == ord('L'):
            p = "asl_features.csv"
            if os.path.exists(p):
                data.load_csv(p)
                clf.fit(data)
            else:
                print("[load] file not found:", p)
        elif k == ord('['):
            label_idx = (label_idx - 1) % len(label_list)
            current_label = label_list[label_idx]
        elif k == ord(']'):
            label_idx = (label_idx + 1) % len(label_list)
            current_label = label_list[label_idx]

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()