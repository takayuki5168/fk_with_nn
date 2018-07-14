(defclass arm-robot
  :super cascaded-link
  :slots (end-coords l1 l2 l3 l4 l5 l6 j1 j2 j3 j4 j5))

(defmethod arm-robot
  (:init ()
	 (let (b)
	   (send-super :init)
	   ;; l7 j7
	   (setq b (make-cube 20 20 20))
	   (send b :locate #f(0 0 10))
	   (send b :set-color :purple)
	   (setq l6 (instance bodyset-link :init (make-cascoords) :bodies (list b) :name 'l6))
	   (setq end-coords (make-cascoords :pos #f(0 0 20)))
	   (send l6 :assoc end-coords)
	   (send l6 :locate #f(0 0 100))
	   ;;l5 j5
	   (setq b (make-cube 20 20 100))
	   (send b :locate #f(0 0 50))
	   (send b :set-color :blue)
	   (setq l5 (instance bodyset-link :init (make-cascoords) :bodies (list b) :name 'l5))
	   (send l5 :assoc l6)
	   (send l5 :locate #f(0 0 100))
	   ;;l4 j4
	   (setq b (make-cube 20 20 100))
	   (send b :locate #f(0 0 50))
	   (send b :set-color :green)
	   (setq l4 (instance bodyset-link :init (make-cascoords) :bodies (list b) :name 'l4))
	   (send l4 :assoc l5)
	   (send l4 :locate #f(0 0 100))
	   ;;l3 j3
	   (setq b (make-cube 20 20 100))
	   (send b :locate #f(0 0 50))
	   (send b :set-color :yellow)
	   (setq l3 (instance bodyset-link :init (make-cascoords) :bodies (list b) :name 'l3))
	   (send l3 :assoc l4)
	   (send l3 :locate #f(0 0 100))
	   ;;l2 j2
	   (setq b (make-cube 20 20 100))
	   (send b :locate #f(0 0 50))
	   (send b :set-color :red)
	   (setq l2 (instance bodyset-link :init (make-cascoords) :bodies (list b) :name 'l2))
	   (send l2 :assoc l3)
	   (send l2 :locate #f(0 0 20))
	   ;; l1 j1
	   (setq b (body+ (make-cube 20 20 20 :pos #f(0 0 10)) (make-cube 300 300 2)))
	   (send b :set-color :white)
	   (setq l1 (instance bodyset-link :init (make-cascoords) :bodies (list b) :name 'l1))
	   (send l1 :assoc l2)
	   
	   ;; joints
	   (setq j1 (instance rotational-joint :init :name 'j1 :parent-link l1 :child-link l2 :axis :x :min -120 :max 120)
		 j2 (instance rotational-joint :init :name 'j2 :parent-link l2 :child-link l3 :axis :x :min -120 :max 120)
		 j3 (instance rotational-joint :init :name 'j3 :parent-link l3 :child-link l4 :axis :x :min -120 :max 120)
		 j4 (instance rotational-joint :init :name 'j4 :parent-link l4 :child-link l5 :axis :x :min -120 :max 120)
		 j5 (instance rotational-joint :init :name 'j5 :parent-link l5 :child-link l6 :axis :x :min -120 :max 120)
		 )
	   (setq links (list l1 l2 l3 l4 l5 l6))
	   (setq joint-list (list j1 j2 j3 j4 j5))
	   (send self :init-ending)
	   self))
  (:end-coords (&rest args) (forward-message-to end-coords args))

  (:get-l1() l1)
  (:get-l2() l2)
  (:get-l3() l3)
  (:get-l4() l4)
  (:get-l5() l5)
  (:get-l6() l6)
  )

(setq *robot* (instance arm-robot :init))
;;(objects (list *robot*))
