The forecasting is regression problem. The implementation is based on xgboost, which is widely popular machine learning models and proved successful in many applications.
+ It include xgboost training & model selection -- fit
+ It include offline evaluation -- predict

Feature extraction:
xgboost is not a time-series model. The features in training & forcasting is a multi-dimensional vector. In the implementation, the element in vectors are:
+ Log-scale price to reduce scale affect
+ N-lookback price window
+ Deriviative feature to capture price changing


