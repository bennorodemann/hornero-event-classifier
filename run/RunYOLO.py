"""Alex: Run YOLO, with two steps"""

import sys
sys.path.append("./run")
# from defaults import YOLO_FOLDER
# from defaults import VIDEOS_ROOT_PATH
import os
import pandas as pd
import MAAP3D


if __name__ == "__main__":
    
    VIDEOS_ROOT_PATH = "/media/alexchan/MSc Lucio/ValidationVideos/videos"
    YOLO_FOLDER = "/media/alexchan/MSc Lucio/ValidationVideos/YOLO/"
    
    VideoPathRoot = os.path.join(VIDEOS_ROOT_PATH,"nest")
    
    AllVids = os.listdir(VideoPathRoot)
    
    MainYOLO = "/media/alexchan/MSc Lucio/ValidationVideos/weights/YOLOOvenbirdapril2026.pt"
    NestedYOLO = "/media/alexchan/MSc Lucio/ValidationVideos/weights/NestedBoxRingYOLO.pt"
    KPPath = ("/media/alexchan/MSc Lucio/ValidationVideos/weights/hrnet.py",
              "/media/alexchan/MSc Lucio/ValidationVideos/weights/hrnet_best_PCK_epoch_300.pth")
    
    
    for w, vid in enumerate(AllVids):
        print("video %s out of %s"%(w+1, len(AllVids)))
        vidname = vid.split(".")[0]
        VideoPath = os.path.join(VideoPathRoot, vid)
        
        # import ipdb;ipdb.set_trace()
        if not os.path.exists(os.path.join(YOLO_FOLDER, "%s_overall.csv"%vidname)):
            mainyolo = MAAP3D.run_yolo(MainYOLO, VideoPath)

            MAAP3D.save_data(mainyolo, os.path.join(YOLO_FOLDER, "%s_overall.csv"%vidname))
        # print("yo")
        mainyolo = MAAP3D.load_data(os.path.join(YOLO_FOLDER, "%s_overall.csv"%vidname))
        
        mainyolo["Class"] = mainyolo["ID"].apply(lambda x: x.split("-")[1])
        
        mainyolo["Class"].unique()
        
        birdonly = mainyolo[mainyolo["Class"] == "bird"]
        
        if not os.path.exists(os.path.join(YOLO_FOLDER, "%s_nested.csv"%vidname)):
            ##nested yolo
            nestedyolo = MAAP3D.run_yolo(NestedYOLO, VideoPath,bbox = birdonly)

            MAAP3D.save_data(nestedyolo, os.path.join(YOLO_FOLDER, "%s_nested.csv"%vidname))

        nestedyolo = MAAP3D.load_data(os.path.join(YOLO_FOLDER, "%s_nested.csv"%vidname))
        Merged = pd.concat([birdonly, nestedyolo], axis = 0)

        MAAP3D.save_data(Merged, os.path.join(YOLO_FOLDER, "%s_bbox.csv"%vidname))
        
        ##runkp
        if not os.path.exists(os.path.join(YOLO_FOLDER, "%s_keypoints.csv"%vidname)):
            mainyolo = MAAP3D.load_data(os.path.join(YOLO_FOLDER, "%s_overall.csv"%vidname))
            mainyolo["Class"] = mainyolo["ID"].apply(lambda x: x.split("-")[1])

            birdonly = mainyolo[mainyolo["Class"] == "bird"]

            OutKP = MAAP3D.run_keypoints_2d(KPPath, VideoPath,bbox = birdonly)
            MAAP3D.save_data(OutKP, os.path.join(YOLO_FOLDER, "%s_keypoints.csv"%vidname))
        
        
        OutDir = "/media/alexchan/MSc Lucio/ValidationVideos/Visualization"
        
        Skeleton = (["bd_shoulder_left","bd_shoulder_right"],["bd_shoulder_left","bd_tail_base"],
        ["bd_shoulder_right","bd_tail_base"],["bd_tail_base","bd_tail_tip"],
        ["lg_anckle_left","lg_foot_left"],["lg_anckle_right","lg_foot_right"],
        ["lg_anckle_left","lg_anckle_right"],["hd_eye_left","hd_eye_right"],
        ["hd_eye_left","hd_bill_base"],["hd_eye_right","hd_bill_base"],
        ["hd_bill_base","hd_bill_tip"])
        
        if not os.path.exists(os.path.join(OutDir, "%s.mp4"%vidname)):
            OutKP = MAAP3D.load_data(os.path.join(YOLO_FOLDER, "%s_keypoints.csv"%vidname))
            Merged = MAAP3D.load_data(os.path.join(YOLO_FOLDER, "%s_bbox.csv"%vidname))

            MAAP3D.visualize_2d(VideoPath, bbox = Merged,key2d=OutKP, save=True,show=False,
                                out_path = os.path.join(OutDir, "%s.mp4"%vidname),skeleton=Skeleton)