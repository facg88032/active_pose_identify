## 人體動態識別

**任何物件都可以有「靜態」和「動態」**的情況，從人類的活動來看，諸如坐著、站著、睡覺…等靜止的行為都屬於靜態，奔跑、跳躍、翻滾…等有位移行動則屬於動態。在辨識兩種情況時，靜態行為相較簡單，因為只需要**考量當前的動作**即可作出判斷，而動態動作需要**考量一段時間序列發生的所有動作**才能進行判斷是相對較困難的議題。

主要架構:

**OpenPose**: 收集影像中人體全身關鍵點的資料

**Keras搭建LSTM**:學習OpenPose 收集的資料進行動態識別



**實際成果**

![](https://github.com/facg88032/active_pose_identify/blob/master/demo/dribble.gif)

![](https://github.com/facg88032/active_pose_identify/blob/master/demo/shoot.gif)

