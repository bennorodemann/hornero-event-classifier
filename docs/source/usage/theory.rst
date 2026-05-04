Theory
======

This page describes the theory step by step process that ``run/classify.py`` uses. This document assumes you are using
the default function values.

Step 1: Loading
---------------

YOLO bounding boxes are loaded and grouped into ``Items`` which are groups of bounding boxes that share the same id. It 
also loads video metadata including frame dimensions and number of frames in video.

Step 2: Slit at Large Gaps
--------------------------

If a bird ``Item`` has a large gap in detections it is assumed that the two sequences are separate birds. In these cases
bird ``Items`` are split into separate items. By default this is ``100`` frames.

Step 3: Fill Missing Frames
---------------------------

All remaining gaps in bird and ring ``Items`` are filled with linearly interpolated bounding boxes. Confidence values
are also interpolated.

The one exception to this step is bird ``Items`` in which the bounding box is within 5 pixels the side for the video 
frame immediately before or after the frame gap. In these cases the bird ``Item`` is split instead.

Step 4: Split at Enter/Leave
----------------------------

When a new bird ``Item`` enters the frame, it can be unclear if it is truly another bird or the same bird being detected
twice. This is further complicated by the fact that sometimes a bird enters the frame, the two ``Items`` switch birds,
and then the other bird leaves. In YOLO this looks like a single bird that stays at the nest for a long time and the
other bird coming to the nest and then immediately leaving, even though in actuality they switched.

To combat this, at the first and last frame of each ``Item``, all other ``Items`` in the same frame are split. This is 
done unless the other item started or will end within ``100`` frames, in which case it is ignored.

Step 5: Remove Low Confidence Birds
-----------------------------------

This simply removes any ``Item`` with an average bounding box confidence below 0.7.

Step 6: Threshold Classification
--------------------------------

The threshold classifier calculates the metrics in ``run/weights.json`` for each individual bounding box in an ``Item``
and then takes the average and multiplies each metric by it's associated weights. These are then all summed together. 
``Items`` that have a final value greater than or equal to the threshold are classified as ringed. 

Values in ``run/weights.json`` where generated using a glm model.

Step 7: Combine Items into events
---------------------------------

Finally, ``Items`` that share the same classification and overlap or are within ``120`` frames of one another are
combined into a single event. Events that last less than ``100`` frames are then removed.
