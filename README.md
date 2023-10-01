# Track_for_robots
This repository is designed for UAV robots. And we follow "TCTrack: Temporal Contexts for Aerial Tracking （CVPR2022) & TCTrack++：Towards Real-World Visual Tracking with Temporal Contexts （TPAMI）" and "Deep Hough Transform for Semantic Line Detection" to complete a task about line detection for UAV robots.
And for TCTrack, there are two main modules. They are online-feature-extraction and similarity refinement. And I would like to use the method of deep hough transform to replace online feature extraction which could have a better result for line detection.

TODO:
1.了解deep hough transform的output的维度，以及与下游input接口的相关性，思考应该如何修改相关接口。（基本的方案：修改上游的输出以与下游的输入相适应（多条线的feature放在一起是否能够产生下游的输入）；修改下游的输入以与上游的输出相适应）（待完成）
2.TCTrack的feature extraction是否有相关的时序信息，在添加deep hough transform来进行feature extraction的时候是否需要增添时序信息（待完成）
