###############################################################################
### Playing rock paper scissor
### Input : Live video of 2 hands playing rock paper scissor
###         Hands should be distinguished: Left and Right
### Output: 2D display of hand keypoint
###         with gesture classification (rock=fist, paper=five, scissor=three/two)
###############################################################################

import cv2
import open3d as o3d
import numpy as np
import mediapipe as mp


# Define default camera intrinsic
img_width  = 640
img_height = 480
intrin_default = {
    'fx': img_width*0.9, # Approx 0.7w < f < w https://www.learnopencv.com/approximate-focal-length-for-webcams-and-cell-phone-cameras/
    'fy': img_width*0.9,
    'cx': img_width*0.5, # Approx center of image
    'cy': img_height*0.5,
    'width': img_width,
    'height': img_height,
}
class DisplayHand:
    def __init__(self, draw3d=False, draw_camera=False, intrin=None, max_num_hands=1, vis=None):
        self.max_num_hands = max_num_hands
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Define kinematic tree linking keypoint together to form skeleton
        self.ktree = [0,  # Wrist
                      0, 1, 2, 3,  # Thumb
                      0, 5, 6, 7,  # Index
                      0, 9, 10, 11,  # Middle
                      0, 13, 14, 15,  # Ring
                      0, 17, 18, 19]  # Little

        # Define color for 21 keypoint
        self.color = [[0, 0, 0],  # Wrist black
                      [255, 0, 0], [255, 60, 0], [255, 120, 0], [255, 180, 0],  # Thumb
                      [0, 255, 0], [60, 255, 0], [120, 255, 0], [180, 255, 0],  # Index
                      [0, 255, 0], [0, 255, 60], [0, 255, 120], [0, 255, 180],  # Middle
                      [0, 0, 255], [0, 60, 255], [0, 120, 255], [0, 180, 255],  # Ring
                      [0, 0, 255], [60, 0, 255], [120, 0, 255], [180, 0, 255]]  # Little
        self.color = np.asarray(self.color)
        self.color_ = self.color / 255  # For Open3D RGB
        self.color[:, [0, 2]] = self.color[:, [2, 0]]  # For OpenCV BGR
        self.color = self.color.tolist()

        ############################
        ### Open3D visualization ###
        ############################
        if draw3d:
            if vis is not None:
                self.vis = vis
            else:
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    width=self.intrin['width'], height=self.intrin['height'])
            self.vis.get_render_option().point_size = 8.0
            joint = np.zeros((21, 3))

            # Draw 21 joints
            self.pcd = []
            for i in range(max_num_hands):
                p = o3d.geometry.PointCloud()
                p.points = o3d.utility.Vector3dVector(joint)
                p.colors = o3d.utility.Vector3dVector(self.color_)
                self.pcd.append(p)

            # Draw 20 bones
            self.bone = []
            for i in range(max_num_hands):
                b = o3d.geometry.LineSet()
                b.points = o3d.utility.Vector3dVector(joint)
                b.colors = o3d.utility.Vector3dVector(self.color_[1:])
                b.lines = o3d.utility.Vector2iVector(
                    [[0, 1], [1, 2], [2, 3], [3, 4],  # Thumb
                     [0, 5], [5, 6], [6, 7], [7, 8],  # Index
                     [0, 9], [9, 10], [10, 11], [11, 12],  # Middle
                     [0, 13], [13, 14], [14, 15], [15, 16],  # Ring
                     [0, 17], [17, 18], [18, 19], [19, 20]])  # Little
                self.bone.append(b)

            # Draw world reference frame
            frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

            # Add geometry to visualize
            self.vis.add_geometry(frame)
            for i in range(max_num_hands):
                self.vis.add_geometry(self.pcd[i])
                self.vis.add_geometry(self.bone[i])

            # Set camera view
            ctr = self.vis.get_view_control()
            ctr.set_up([0, -1, 0])  # Set up as -y axis
            ctr.set_front([0, 0, -1])  # Set to looking towards -z axis
            ctr.set_lookat([0.5, 0.5, 0])  # Set to center of view
            ctr.set_zoom(1)

            if draw_camera:
                # Remove previous frame
                self.vis.remove_geometry(frame)
                # Draw camera reference frame
                frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
                # Draw camera frustum
                self.camera = DisplayCamera(self.vis, self.intrin)
                frustum = self.camera.create_camera_frustum()
                # Draw 2D image plane in 3D space
                self.mesh_img = self.camera.create_mesh_img()
                # Add geometry to visualize
                self.vis.add_geometry(frame)
                self.vis.add_geometry(frustum)
                self.vis.add_geometry(self.mesh_img)
                # Reset camera view
                self.camera.reset_view()

    def draw_game_rps(self, img, param):
        img_height, img_width, _ = img.shape

        # Init result of 2 hands to none
        res = [None, None]

        # Loop through different hands
        for j, p in enumerate(param):
            # Only allow maximum of two hands
            if j > 1:
                break

            if p['class'] is not None:
                # Loop through keypoint for each hand
                for i in range(21):
                    x = int(p['keypt'][i, 0])
                    y = int(p['keypt'][i, 1])
                    if x > 0 and y > 0 and x < img_width and y < img_height:
                        # Draw skeleton
                        start = p['keypt'][self.ktree[i], :]
                        x_ = int(start[0])
                        y_ = int(start[1])
                        if x_ > 0 and y_ > 0 and x_ < img_width and y_ < img_height:
                            cv2.line(img, (x_, y_), (x, y), self.color[i], 2)

                        # Draw keypoint
                        cv2.circle(img, (x, y), 5, self.color[i], -1)

                # Label gesture
                text = None
                if p['gesture'] == 'fist':
                    text = 'rock'
                elif p['gesture'] == 'five':
                    text = 'paper'
                elif (p['gesture'] == 'three') or (p['gesture'] == 'yeah'):
                    text = 'scissor'
                res[j] = text

                # Label result
                if text is not None:
                    x = int(p['keypt'][0, 0]) - 30
                    y = int(p['keypt'][0, 1]) + 40
                    cv2.putText(img, '%s' % (text.upper()), (x, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)  # Red

        # Determine winner
        text = None
        winner = None
        if res[0] == 'rock':
            if res[1] == 'rock':
                text = 'Tie'
            elif res[1] == 'paper':
                text = 'Paper wins'; winner = 1
            elif res[1] == 'scissor':
                text = 'Rock wins'; winner = 0
        elif res[0] == 'paper':
            if res[1] == 'rock':
                text = 'Paper wins'; winner = 0
            elif res[1] == 'paper':
                text = 'Tie'
            elif res[1] == 'scissor':
                text = 'Scissor wins'; winner = 1
        elif res[0] == 'scissor':
            if res[1] == 'rock':
                text = 'Rock wins'; winner = 1
            elif res[1] == 'paper':
                text = 'Scissor wins'; winner = 0
            elif res[1] == 'scissor':
                text = 'Tie'

        # Label gesture
        if text is not None:
            size = cv2.getTextSize(text.upper(),
                                   # cv2.FONT_HERSHEY_SIMPLEX, 2, 2)[0]
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            x = int((img_width - size[0]) / 2)
            cv2.putText(img, text.upper(),
                        # (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
                        (x, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Draw winner text
        if winner is not None:
            x = int(param[winner]['keypt'][0, 0]) - 30
            y = int(param[winner]['keypt'][0, 1]) + 80
            cv2.putText(img, 'WINNER', (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # Yellow

        return img


class MediaPipeHand:
    def __init__(self, static_image_mode=True, max_num_hands=1, intrin=None):
        self.max_num_hands = max_num_hands
        if intrin is None:
            self.intrin = intrin_default
        else:
            self.intrin = intrin

        # Access MediaPipe Solutions Python API
        mp_hands = mp.solutions.hands
        self.pipe = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Define hand parameter
        self.param = []
        for i in range(max_num_hands):
            p = {
                'keypt': np.zeros((21, 2)),  # 2D keypt in image coordinate (pixel)
                'joint': np.zeros((21, 3)),  # 3D joint in relative coordinate
                'joint_3d': np.zeros((21, 3)),  # 3D joint in camera coordinate (m)
                'class': None,  # Left / right / none hand
                'score': 0,  # Probability of predicted handedness (always>0.5, and opposite handedness=1-score)
                'angle': np.zeros(15),  # Flexion joint angles in degree
                'gesture': None,  # Type of hand gesture
                'fps': -1,  # Frame per sec
            }
            self.param.append(p)

    def result_to_param(self, result, img):
        # Convert mediapipe result to my own param
        img_height, img_width, _ = img.shape

        # Reset param
        for p in self.param:
            p['class'] = None

        if result.multi_hand_landmarks is not None:
            # Loop through different hands
            for i, res in enumerate(result.multi_handedness):
                if i > self.max_num_hands - 1: break  # Note: Need to check if exceed max number of hand
                self.param[i]['class'] = res.classification[0].label
                self.param[i]['score'] = res.classification[0].score

            # Loop through different hands
            for i, res in enumerate(result.multi_hand_landmarks):
                if i > self.max_num_hands - 1: break  # Note: Need to check if exceed max number of hand
                # Loop through 21 landmark for each hand
                for j, lm in enumerate(res.landmark):
                    self.param[i]['keypt'][
                        j, 0] = lm.x * img_width  # Convert normalized coor to pixel [0,1] -> [0,width]
                    self.param[i]['keypt'][
                        j, 1] = lm.y * img_height  # Convert normalized coor to pixel [0,1] -> [0,height]

                    self.param[i]['joint'][j, 0] = lm.x
                    self.param[i]['joint'][j, 1] = lm.y
                    self.param[i]['joint'][j, 2] = lm.z

                # Convert relative 3D joint to angle
                self.param[i]['angle'] = self.convert_3d_joint_to_angle(self.param[i]['joint'])
                # Convert relative 3D joint to actual 3D joint in camera coordinate
                self.convert_relative_to_actual_3d_joint(self.param[i], self.intrin)

        return self.param

    def convert_3d_joint_to_angle(self, joint):
        # Get direction vector of bone from parent to child
        v1 = joint[[0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19], :]  # Parent joint
        v2 = joint[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], :]  # Child joint
        v = v2 - v1  # [20,3]
        # Normalize v
        v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

        # Get angle using arcos of dot product
        angle = np.arccos(np.einsum('nt,nt->n',
                                    v[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 18], :],
                                    v[[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19], :]))  # [15,]

        return np.degrees(angle)  # Convert radian to degree

    def convert_relative_to_actual_3d_joint(self, param, intrin):

        # De-normalized 3D hand joint
        param['joint_3d'][:, 0] = param['joint'][:, 0] * intrin['width'] - intrin['cx']
        param['joint_3d'][:, 1] = param['joint'][:, 1] * intrin['height'] - intrin['cy']
        param['joint_3d'][:, 2] = param['joint'][:, 2] * intrin['width']

        # Assume average depth is fixed at 0.6 m (works best when the hand is around 0.5 to 0.7 m from camera)
        Zavg = 0.6
        # Average focal length of fx and fy
        favg = (intrin['fx'] + intrin['fy']) * 0.5
        # Compute scaling factor S
        S = favg / Zavg
        # Uniform scaling
        param['joint_3d'] /= S

        # Estimate wrist depth using similar triangle
        D = 0.08  # Note: Hardcode actual dist btw wrist and index finger MCP as 0.08 m
        # Dist btw wrist and index finger MCP keypt (in 2D image coor)
        d = np.linalg.norm(param['keypt'][0] - param['keypt'][9])
        # d/f = D/Z -> Z = D/d*f
        Zwrist = D / d * favg
        # Add wrist depth to all joints
        param['joint_3d'][:, 2] += Zwrist

    def forward(self, img):
        # Preprocess image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Extract result
        result = self.pipe.process(img)

        # Convert result to my own param
        param = self.result_to_param(result, img)

        return param


#############################################################
### Simple gesture recognition from joint angle using KNN ###
#############################################################
class GestureRecognition:
    def __init__(self, mode='train'):
        super(GestureRecognition, self).__init__()

        # 11 types of gesture 'name':class label
        self.gesture = {
            'fist': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6,
            'rock': 7, 'spiderman': 8, 'yeah': 9, 'ok': 10,
        }

        if mode == 'train':
            # Create .csv file to log training data
            self.file = open('./gesture_train.csv', 'a+')
        elif mode == 'eval':
            # Load training data
            file = np.genfromtxt('./gesture_train.csv', delimiter=',')
            # Extract input joint angles
            angle = file[:, :-1].astype(np.float32)
            # Extract output class label
            label = file[:, -1].astype(np.float32)
            # Use OpenCV KNN
            self.knn = cv2.ml.KNearest_create()
            self.knn.train(angle, cv2.ml.ROW_SAMPLE, label)

    def train(self, angle, label):
        # Log training data
        data = np.append(angle, label)  # Combine into one array
        np.savetxt(self.file, [data], delimiter=',', fmt='%f')

    def eval(self, angle):
        # Use KNN for gesture recognition
        data = np.asarray([angle], dtype=np.float32)
        ret, results, neighbours, dist = self.knn.findNearest(data, 3)
        idx = int(results[0][0])  # Index of class label

        return list(self.gesture)[idx]  # Return name of class label


# Load mediapipe hand class
pipe = MediaPipeHand(static_image_mode=False, max_num_hands=2)

# Load display class
disp = DisplayHand(max_num_hands=2)

# Start video capture
cap = cv2.VideoCapture(0) # By default webcam is index 0

# Load gesture recognition class
gest = GestureRecognition(mode='eval')

counter = 0
while cap.isOpened():
    ret, img = cap.read()
    if not ret:
        break

    # Flip image for 3rd person view
    img = cv2.flip(img, 1)

    # To improve performance, optionally mark image as not writeable to pass by reference
    img.flags.writeable = False

    # Feed forward to extract keypoint
    param = pipe.forward(img)
    # Evaluate gesture for all hands
    for p in param:
        if p['class'] is not None:
            p['gesture'] = gest.eval(p['angle'])
    img.flags.writeable = True

    # Display keypoint and result of rock paper scissor game
    cv2.imshow('Game: Rock Paper Scissor', disp.draw_game_rps(img.copy(), param))

    key = cv2.waitKey(1)
    if key==27:
        break

pipe.pipe.close()
cap.release()