# FaceID

## Face Detection and Recognition

___Detection___ using haarscascade Classifier
* front faces using haarcascade_frontalface_default.xml
* side faces using haarcascade_profileface.xml
* eye  - check for eyes inside bounded faces


___Recognition___ using Local Binary Pattern Histogram (LBPH) Face Recognition.

`train_images/` contains raw images used for training the model
`processed_images/` contains images that have been processed for model training
`test_images/` contains images used to test the model