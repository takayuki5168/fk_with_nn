# Forward Kinematics with Deep Neural Network

## Demo
![FK](https://github.com/takayuki5168/fk_with_nn/gif/random-fk.gif)
![IK](https://github.com/takayuki5168/fk_with_nn/gif/ik-with-nn.gif)

## Architecture
Input : all joint angle

Output : all joint position (not only end effector of arm)

By using BP between result of FP and reference, we can solve IK