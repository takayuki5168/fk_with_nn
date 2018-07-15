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
(setq *ref-coords* (make-coords))
(send *ref-coords* :locate #f(0 186.603 363.205))
(send *robot* :inverse-kinematics *ref-coords* :link-list (send *robot* :link-list (send *robot* :end-coords :parent)) :move-target (send *robot* :end-coords) :translation-axis t :rotation-axis t)
(setq *avoid-position* (list (make-coords) (make-coords) (make-coords) (make-coords) (make-coords) (make-coords)))
(send (elt *avoid-position* 3) :locate #f(0 136.603 263.205))
(send (elt *avoid-position* 5) :locate (send *ref-coords* :worldpos))
(objects (list *robot*
	       (elt *avoid-position* 0)
	       (elt *avoid-position* 1)
	       (elt *avoid-position* 2)
	       (elt *avoid-position* 3)
	       (elt *avoid-position* 4)
	       (elt *avoid-position* 5)
	       ))

;; init functions
(defun check-zero-vec (a)
  (and
   (eq (elt a 0) 0.0)
   (eq (elt a 1) 0.0)
   (eq (elt a 2) 0.0)
   )
  )

(defun bool2float (a)
  (if (eq a t) 1.0 0.0)
  )

(defun valid-vec (a)
  (float-vector
   (bool2float (not (check-zero-vec (send (elt a 0) :worldpos))))
   (bool2float (not (check-zero-vec (send (elt a 1) :worldpos))))
   (bool2float (not (check-zero-vec (send (elt a 2) :worldpos))))
   (bool2float (not (check-zero-vec (send (elt a 3) :worldpos))))
   (bool2float (not (check-zero-vec (send (elt a 4) :worldpos))))
   (bool2float (not (check-zero-vec (send (elt a 5) :worldpos))))      
   )
  )

(do-until-key
 (x::window-main-one) 
 )

(let ()
  (do-until-key
    ;; publish angle vector
    (setq msg (instance std_msgs::Float64MultiArray :init))
    (send msg :data (send *robot* :angle-vector))
    (ros::publish "angle_vector" msg)
    
    ;; draw
    (send *irtviewer* :draw-objects)    
    
    ;; send req
    (setq req (instance fk_with_nn::IkRequest :init))
    (send req :pos_x (float-vector (elt (send (elt *avoid-position* 0) :worldpos) 0)
				   (elt (send (elt *avoid-position* 1) :worldpos) 0)
				   (elt (send (elt *avoid-position* 2) :worldpos) 0)
				   (elt (send (elt *avoid-position* 3) :worldpos) 0)
				   (elt (send (elt *avoid-position* 4) :worldpos) 0)
				   (elt (send (elt *avoid-position* 5) :worldpos) 0)))
    (send req :pos_y (float-vector (elt (send (elt *avoid-position* 0) :worldpos) 1)
				   (elt (send (elt *avoid-position* 1) :worldpos) 1)
				   (elt (send (elt *avoid-position* 2) :worldpos) 1)
				   (elt (send (elt *avoid-position* 3) :worldpos) 1)
				   (elt (send (elt *avoid-position* 4) :worldpos) 1)
				   (elt (send (elt *avoid-position* 5) :worldpos) 1)))
    (send req :pos_z (float-vector (elt (send (elt *avoid-position* 0) :worldpos) 2)
				   (elt (send (elt *avoid-position* 1) :worldpos) 2)
				   (elt (send (elt *avoid-position* 2) :worldpos) 2)
				   (elt (send (elt *avoid-position* 3) :worldpos) 2)
				   (elt (send (elt *avoid-position* 4) :worldpos) 2)
				   (elt (send (elt *avoid-position* 5) :worldpos) 2)))
    (send req :data (valid-vec *avoid-position*))

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
