# Unet

###
Hyunbin  
colab 이슈땜에 어제 저녁에 학습 시작됐는디 다들 어차피 이미 backbone 써서 하고있길래 주어진 baseline 어떻게 발전 못시키나 함 시도해봄  

train set -> [:6500]  
valid set -> [6500:]  

Activation:  
Swish 쓸 경우 오히려 성능 떨어뜨림  
ReLU가 가장 성능 나았음  

기법:  
Deep Supervision 쓸 경우 성능 조금 좋아졌음  
Residual Attention + Deep Supervision 해보려했는데 channel 안맞는 문제로 실패 (내 실수)  

Loss:  
Dice+BCE 쓸 경우 별로 효과가 없었음  
MixedLoss(Focal, Dice)가 가장 성능 좋았음  

Deep SUpervision + MixedLoss 써서 valid inference 돌려봤을 때 dice score 0.69 나와서 함 제출해봤는디 제출 점수는 0.60 나옴.  

결론->baseline은 ㄹㅇ 버리는게 맞는듯  
###
