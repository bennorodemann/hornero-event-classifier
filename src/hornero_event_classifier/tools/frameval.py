import time
from abc import ABC, abstractmethod
from collections import defaultdict
import csv
from dataclasses import dataclass, field

# TODO: make run function
# TODO: make cmd tool
# TODO: comment
# TODO: document
# TODO: write output
# TODO: optimize? (.5 s: good, .2 s: excellent)

class NearInt:
    def __init__(self, val: int, buffer: int):
        self.val = val
        self.buffer = buffer
        self.upper = val + buffer
        self.lower = val - buffer

    def __eq__(self, value):
        return value < self.upper and value > self.lower
    
class FrameInfo:
    def __init__(self):
        self.YOLO_birds = 0
        self.YOLO_ringed = 0
        self.YOLO_ringed_mud = 0
        self.YOLO_unringed_mud = 0

        self.BORIS_birds = 0
        self.BORIS_ringed = 0
        self.BORIS_ringed_mud = 0
        self.BORIS_unringed_mud = 0

        self.buffer_frame = False
        self.ring_valid = False
        self.mud_valid = False

    @property
    def empty(self):
        return (self.YOLO_birds == 0) and (self.BORIS_birds == 0)

    def as_dict(self):
        return {"YOLO_birds": self.YOLO_birds,
                "BORIS_birds": self.BORIS_birds}

    def add_YOLO_data(self, ringed, mud):
        self.YOLO_birds += 1
        mud = 1 if mud else 0
        if ringed == "ring":
            self.YOLO_ringed += 1
            self.YOLO_ringed_mud += mud
        else:
            self.YOLO_unringed_mud += mud

    def add_BORIS_data(self, ringed, mud):
        self.BORIS_birds += 1
        mud = 1 if mud else 0
        if ringed == "ring":
            self.BORIS_ringed += 1
            self.BORIS_ringed_mud += mud
        else:
            self.BORIS_unringed_mud += mud

    @property
    def YOLO_unringed(self):
        return self.YOLO_birds - self.YOLO_ringed
    
    @property
    def BORIS_unringed(self):
        return self.BORIS_birds - self.BORIS_ringed
    
    @property
    def YOLO_ringed_no_mud(self):
        return self.YOLO_ringed - self.YOLO_ringed_mud
    
    @property
    def YOLO_unringed_no_mud(self):
        return self.YOLO_unringed - self.YOLO_unringed_mud
    
    @property
    def BORIS_ringed_no_mud(self):
        return self.BORIS_ringed - self.BORIS_ringed_mud
    
    @property
    def BORIS_unringed_no_mud(self):
        return self.BORIS_unringed - self.BORIS_unringed_mud
    
class ValidationError(Exception):
    ...
    
@dataclass(slots=True)
class Validator(ABC):
    TP: int = field(default=0, init=False)
    TN: int = field(default=0, init=False)
    FP: int = field(default=0, init=False)
    FN: int = field(default=0, init=False)
    frame_counter: int = field(default=0, init=False, repr=False)

    @abstractmethod
    def validate(self, frame: FrameInfo):
        ...

    def total(self):
        return self.TP + self.TN + self.FP + self.FN
    
    def _write_data(self, TP: int = 0, FP: int = 0, FN: int = 0, TN: int = 0):
        self.TP += TP
        self.FP += FP
        self.FN += FN
        self.TN += TN

class BirdValidator(Validator):
    def validate(self, frame: FrameInfo):
        if not frame.empty:
            TP = min(frame.YOLO_birds, frame.BORIS_birds)
            dif = frame.YOLO_birds - frame.BORIS_birds
            if dif == 0:
                frame.ring_valid = True
                FP = 0
                FN = 0
            elif dif < 0:
                FN = abs(dif)
                FP = 0
            else:
                FP = dif
                FN = 0
            if (TP + FN + FP) != max(frame.YOLO_birds, frame.BORIS_birds):
                raise ValidationError(f"Bird validation was unsuccessful. Calculated {TP + FP + FN} validations but should have got {max(frame.YOLO_birds, frame.BORIS_birds)}")
            self.frame_counter += 1
            self._write_data(TP=TP, FP=FP, FN=FN)

    def calculate_TN(self, video_length: int):
        self.TN = video_length - self.frame_counter
        
    
class BufferValidator(BirdValidator):
    def __init__(self):
        super().__init__()
        self.banned_counter: int = 0

    def validate(self, frame: FrameInfo):
        if not frame.buffer_frame:
            super().validate(frame)
        else:
            self.banned_counter += 1

    def calculate_TN(self, video_length):
        super().calculate_TN(video_length)
        self.TN -= self.banned_counter

    
class RingValidator(Validator):
    def validate(self, frame: FrameInfo):
        if frame.ring_valid:
            TP = min(frame.YOLO_ringed, frame.BORIS_ringed)
            TN = min(frame.YOLO_unringed, frame.BORIS_unringed)
            FP = frame.YOLO_ringed - TP
            FN = frame.YOLO_unringed - TN
            if FP == 0 and FN == 0:
                frame.mud_valid = True
            self._write_data(TP=TP, FP=FP, FN=FN, TN=TN)
            self.frame_counter += 1
    
class MudValidator(Validator):
    def validate(self, frame: FrameInfo):
        if frame.mud_valid:
            ringed_TP = min(frame.YOLO_ringed_mud, frame.BORIS_ringed_mud)
            ringed_TN = min(frame.YOLO_ringed_no_mud, frame.BORIS_ringed_no_mud)
            ringed_FP = frame.YOLO_ringed_mud - ringed_TP
            ringed_FN = frame.YOLO_ringed_no_mud - ringed_TN

            unringed_TP = min(frame.YOLO_unringed_mud, frame.BORIS_unringed_mud)
            unringed_TN = min(frame.YOLO_unringed_no_mud, frame.BORIS_unringed_no_mud)
            unringed_FP = frame.YOLO_unringed_mud - unringed_TP
            unringed_FN = frame.YOLO_unringed_no_mud - unringed_TN
            
            self._write_data(TP = ringed_TP + unringed_TP,
                             TN = ringed_TN + unringed_TN,
                             FN = ringed_FN + unringed_FN,
                             FP = ringed_FP + unringed_FP)
            self.frame_counter += 1

def write(filename, frames: dict[int, FrameInfo]):
    with open(filename, "w", newline='') as file:
        csv_writer = csv.DictWriter(file, ("frame", "YOLO_birds", "BORIS_birds", "birds_TP"))
        csv_writer.writeheader()
        for frame in frames:
            frame_data = frames[frame]
            if (not frame_data.empty):
                row = frame_data.as_dict()
                row["frame"] = frame
                row["birds_TP"] = min([frame_data.YOLO_birds, frame_data.BORIS_birds])
                csv_writer.writerow(row)

    
def run():
    filename = "AI_accuracy/DBs_YOLO/pYOLOevents/n1_d1_c2_3_events.csv"
    frames = defaultdict(FrameInfo)
    bird = BirdValidator()
    buf = BufferValidator()
    ring = RingValidator()
    mud = MudValidator()
    vid_length: int = 71040
    t1 = time.time()
    with open(filename, "r", newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            frames[int(row["frame"])].add_YOLO_data(row["subject"], row["mud"] == 'True')
    with open("AI_accuracy/DB_BORIS.csv", "r", newline='') as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            if row["video_id"] == "n1_d1_c2_3":
                start: int = int(row["start_frame"])
                end: int = int(row["end_frame"])
                for frame in range(start, end+1):
                    frames[frame].add_BORIS_data(row["subject"], row["mud"] == "TRUE")
                for frame in range(start-BUFFER_FRAMES, start+BUFFER_FRAMES+1):
                    if frame >= 0 and frame < vid_length:
                        frames[frame].buffer_frame = True
                for frame in range(end-BUFFER_FRAMES, end+BUFFER_FRAMES+1):
                    if frame >= 0 and frame < vid_length:
                        frames[frame].buffer_frame = True
    write("AI_accuracy/DBs_YOLO/YOLOvalid/n1_d1_c2_3_bbox2.csv", frames)
    for frame in frames.values():
        bird.validate(frame)
        buf.validate(frame)
        ring.validate(frame)
        mud.validate(frame)
    bird.calculate_TN(vid_length)
    buf.calculate_TN(vid_length)
    with open('t.csv', 'w', newline='') as f:
        dw = csv.DictWriter(f, ("n"))
        dw.writeheader()
        dw.writerows({'n': f} for f,fi in frames.items() if fi.buffer_frame)
    print(time.time()-t1)
    print(bird)
    print(bird.total())
    print(buf)
    print(buf.total())
    print(ring)
    print(ring.total())
    print(mud)
    print(mud.total())

BUFFER_FRAMES: int = 10
if __name__ == "__main__":
    run()

    


