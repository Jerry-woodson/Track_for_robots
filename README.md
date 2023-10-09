# Track_for_robots
This repository is designed for UAV robots. We follow <b>"TCTrack: Temporal Contexts for Aerial Tracking （CVPR2022) & TCTrack++：Towards Real-World Visual Tracking with Temporal Contexts （TPAMI）"</b> and <b>"Deep Hough Transform for Semantic Line Detection"</b>. 
<br>And for TCTrack, there are two main modules which are online-feature-extraction and similarity map refinement. And we would like to replace online-feature extraction with deep hough transform to have a better result for line detection of UAV robots.
<br>TODO:
<br>1.了解上游任务feature extraction的output的维度，同时熟悉下游input的维度，调整相应的接口（正在进行）
<br>2.了解online-feature-extraction是否有时序信息，是否需要在deep hough transform上面增加时序信息(待完成)
<br>3.制作用于train和test的数据集（待完成）
<br>4.思考文章的逻辑结构（待完成）--已完成Abstract和Introduction部分的内容

