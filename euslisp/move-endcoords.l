(load "arm-robot.l")

(setq *start-coords* (make-coords))
(send *start-coords* :locate #f(100 0 0))

(setq *end-coords* (make-coords))
(send *end-coords* :locate #f(300 300 0))

(setq *now-coords* *start-coords*)

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

(let (vector-to-goal normed-vector scaled-vector next-pos)
  (dotimes (i 10)
    (setq vector-to-goal (sub-vec (send *end-coords* :pos) (send *now-coords* :pos)))
    (setq normed-vector (norm-vec vector-to-goal))
    (setq scaled-vector (scale-vec vector-to-goal (/ 10 normed-vector)))
    (setq next-pos (sum-vec (send *now-coords* :pos) scaled-vector))

    (print next-pos)
    (send *now-coords* :translate scaled-vector)
    )
  )
