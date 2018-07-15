(load "arm-robot.l")

;; init
(ros::load-ros-manifest "roseus")
(ros::load-ros-manifest "fk_with_nn")
(ros::load-ros-manifest "std_msgs")
(ros::roseus "fk_client_move")

;; service and topic publisher
(ros::wait-for-service "ik")
(ros::advertise "angle_vector" std_msgs::Float64MultiArray 1)

;; init variable
(setq *start-coords* (make-coords))
(send *start-coords* :locate #f(0 186.603 363.205))
(setq *end-coords* (make-coords))
(send *end-coords* :locate #f(0 -100 -300)) ;;-186.603 -100)) ;;363.205))
(send *robot* :inverse-kinematics *start-coords* :link-list (send *robot* :link-list (send *robot* :end-coords :parent)) :move-target (send *robot* :end-coords) :translation-axis t :rotation-axis t)
(setq *now-coords* (send *robot* :end-coords))
(setq *ref-pos* (make-sphere 10))
(objects (list *robot* *ref-pos* *start-coords* *end-coords*))

;; init functions
(defun sum-vec (a b)
  (float-vector
   (+ (elt a 0) (elt b 0))
   (+ (elt a 1) (elt b 1))
   (+ (elt a 2) (elt b 2))
   )
  )

(defun sub-vec (a b)
  (float-vector
   (- (elt a 0) (elt b 0))
   (- (elt a 1) (elt b 1))
   (- (elt a 2) (elt b 2))
   )
  )

(defun norm-vec (a)
  (sqrt
   (+ (* (elt a 0) (elt a 0))
      (* (elt a 1) (elt a 1))
      (* (elt a 2) (elt a 2))
      )
   )
  )

(defun scale-vec (a b)
  (float-vector
   (* (elt a 0) b)
   (* (elt a 1) b)
   (* (elt a 2) b)
   )
  )

;;(do-until-key
;; (x::window-main-one) 
;; )

(let (vector-to-goal normed-vector scaled-vector next-pos)
  (do-until-key
    ;; publish angle vector
    (setq msg (instance std_msgs::Float64MultiArray :init))
    (send msg :data (send *robot* :angle-vector))
    (ros::publish "angle_vector" msg)
    
    ;; calc next pos of end effector and draw
    (setq *now-coords* (send *robot* :end-coords))
    (setq vector-to-goal (sub-vec (send *end-coords* :pos) (send *now-coords* :worldpos)))
    (setq normed-vector (norm-vec vector-to-goal))
    (setq scaled-vector (scale-vec vector-to-goal (/ 50. normed-vector)))
    (setq next-pos (sum-vec (send *now-coords* :worldpos) scaled-vector))
    (send *ref-pos* :reset-coords)
    (send *ref-pos* :locate (float-vector 50 (elt next-pos 1) (elt next-pos 2)))
    (send *irtviewer* :draw-objects)    
    
    ;; send req
    (setq req (instance fk_with_nn::IkRequest :init))
    (send req :pos_x (float-vector (elt (send (send *robot* :get-l2) :worldpos) 0)
				   (elt (send (send *robot* :get-l3) :worldpos) 0)
				   (elt (send (send *robot* :get-l4) :worldpos) 0)
				   (elt (send (send *robot* :get-l5) :worldpos) 0)
				   (elt (send (send *robot* :get-l6) :worldpos) 0)
				   (elt next-pos 0)))
    (send req :pos_y (float-vector (elt (send (send *robot* :get-l2) :worldpos) 1)
				   (elt (send (send *robot* :get-l3) :worldpos) 1)
				   (elt (send (send *robot* :get-l4) :worldpos) 1)
				   (elt (send (send *robot* :get-l5) :worldpos) 1)
				   (elt (send (send *robot* :get-l6) :worldpos) 1)
				   (elt next-pos 1)))
    (send req :pos_z (float-vector (elt (send (send *robot* :get-l2) :worldpos) 2)
				   (elt (send (send *robot* :get-l3) :worldpos) 2)
				   (elt (send (send *robot* :get-l4) :worldpos) 2)
				   (elt (send (send *robot* :get-l5) :worldpos) 2)
				   (elt (send (send *robot* :get-l6) :worldpos) 2)
				   (elt next-pos 2)))

    ;; get res and send angle vector to robot
    (setq res (ros::service-call "ik" req t))
    (print (send res :joint_angle))
    (send *robot* :angle-vector (send res :joint_angle))

    ;; wait
    (print "wait 0.1s")
    (setq before (ros::time-now))
    (while (> 0.1 (send (ros::time- (ros::time-now) before) :to-sec))
      (x::window-main-one)
      )
    )
  )

(exit)
