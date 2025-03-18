import ai2thor.controller
import cv2
import numpy as np

# Initialize AI2-THOR Environment
controller = ai2thor.controller.Controller(
    agentMode="default",
    # visibilityDistance=1.0,
    scene="FloorPlan1",
    # renderDepthImage=True,
    # renderInstanceSegmentation=True,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    renderSemanticSegmentation=False,
    gridSize=0.25,
    width=640,
    height=480
)
controller.reset()
controller.step(action="Initialize", gridSize=0.25)

# first frame
e = controller.step(action="Pass")
f = e.frame
gray_prev = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

# Lucas-Kanade Optical Flow parameters
f_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Detect key points in the first frame
prev_points = cv2.goodFeaturesToTrack(gray_prev, mask=None, **f_params)

def compute_foe(points_old, points_new):
    """Compute the Focus of Expansion (FOE) from motion vectors."""
    flow_vectors = points_new - points_old
    A = np.column_stack((flow_vectors[:, 0], np.ones(flow_vectors.shape[0])))
    b = flow_vectors[:, 1]
    slope, intercept = np.linalg.lstsq(A, b, rcond=None)[0]
    foe_x = -intercept / slope
    foe_y = intercept
    return (foe_x, foe_y)

def generate_potential_field(foe, goal):
    """Generate a potential field where obstacles repel and the target attracts."""
    goal_vector = np.array(goal) - np.array(foe)
    goal_vector = goal_vector / np.linalg.norm(goal_vector)  # Normalize
    return goal_vector

def move_towards_goal(foe, goal):
    """Move the agent based on the potential field direction."""
    move_vector = generate_potential_field(foe, goal)
    
    if move_vector[0] > 0:
        controller.step(action="MoveRight")
    else:
        controller.step(action="MoveLeft")
    
    if move_vector[1] > 0:
        controller.step(action="MoveAhead")
    else:
        controller.step(action="MoveBack")

target = (-600, -600)

while True:
    e = controller.step(action="Pass")
    f = e.frame
    gray_next = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)

    next_points, status, err = cv2.calcOpticalFlowPyrLK(gray_prev, gray_next, prev_points, None, **lk_params)
    
    if next_points is not None:
        good_new = next_points[status == 1]
        good_old = prev_points[status == 1]
        
        foe = compute_foe(good_old, good_new)
        move_towards_goal(foe, target)
        
        gray_prev = gray_next.copy()
        prev_points = good_new.reshape(-1, 1, 2)
    
        for new, old in zip(good_new, good_old):
            a, b = new.ravel()
            c, d = old.ravel()
            gray_next = cv2.circle(gray_next, (int(a), int(b)), 3, (0, 0, 255), -1)
            gray_next = cv2.line(gray_next, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
        
        cv2.imshow("Optical Flow Navigation", gray_next)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
controller.stop()