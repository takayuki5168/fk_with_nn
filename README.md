# 5DOF arm robot calculating Forward Kinematics with Deep Neural Network

## Demo
### Forward Kinematics
![FK](https://github.com/takayuki5168/fk_with_nn/blob/master/gif/random-fk.gif)

### Inverse Kinematics
![IK](https://github.com/takayuki5168/fk_with_nn/blob/master/gif/ik-with-nn.gif)

## Network Architecture
Input : all joint angle

Output : all joint position (not only end effector of arm)

### How to solve IK with network representing FK
By using BP between result of FP and reference, we can solve IK