import mediapipe as mp

## predefined mediapipe landmarks for different face parts
## eyes
eye_l = [362,398,384,385,386,387,388,466,263,249,390,373,374,380,381,382]# list(mp.solutions.face_mesh.FACEMESH_LEFT_EYE)
eye_r = [33,246,161,160,159,158,157,173,133,155,154,153,145,144,163,7]#list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYE)
# both_eyes = eye_l + eye_r

## iris
both_irises = list(mp.solutions.face_mesh.FACEMESH_IRISES)#+list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)

## lips
mp_lips = list(mp.solutions.face_mesh.FACEMESH_LIPS)


## face_bottom
face_bottom = [93,137,123,101,126,129,165,391,358,355,330,352,366,323,361,288,397,365,379,378,400,377,152,
                    148,176,150,136,172,58,132]

# mp_nose = list(mp.solutions.face_mesh.FACEMESH_NOSE)
# mp_nose = [6,351,412,343,437,420,360,344,438,309,250,462,370,94,
#            141,242,20,79,218,115,131,198,217,114,188,122,6]

mp_nose = [6,351,412,343,437,420,360,344,326,99,115,131,198,217,114,188,122,6]

# ## eyebrows
# both_brows = list(mp.solutions.face_mesh.FACEMESH_LEFT_EYEBROW)+list(mp.solutions.face_mesh.FACEMESH_RIGHT_EYEBROW)

## upper eyes
mp_eye_Upr_R = [226,113,225,224,223,222,221,189,244,243,133,173,157,158,159,160,161,246,33,130]
mp_eye_Upr_L = [464,413,441,442,443,444,445,342,446,359,263,466,388,387,386,385,384,398,362,463]

## eyebrows
mp_brow_R = [226,113,225,224,223,222,221,189, 193,168,107,66,105,63,70,156]
mp_brow_L = [413,441,442,443,444,445,342,446, 265,372,383,300,293,334,296,336,168,417]

# ## narrow
# mp_eye_Dwn_R = [226,130,33,7,163,144,145,153,154,155,133,243,244,233,232,231,230,229,228,31]
# mp_eye_Dwn_L = [464,463,362,382,381,380,374,373,390,249,263,359,446,261,448,449,450,451,452,453]

## broad
mp_eye_Dwn_R = [226,130,33,7,163,144,145,153,154,155,133,243,244,128,121,120,119,118,117,111,35]
mp_eye_Dwn_L = [464,463,362,382,381,380,374,373,390,249,263,359,446,340,346,347,348,349,350,357]

## lips
# mp_lips = [270, 317, 81, 91, 37, 84, 269, 321, 318, 312, 415, 17, 61, 78, 0, 82, 314, 178, 267,
#            61, 14, 88, 185, 405, 13, 324, 409, 146, 87, 78, 95, 311, 39, 40, 402, 191, 80, 310, 181, 375]
mp_lips = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]

## chin
mp_chin = [136,135,43,106,182,83,18,313,406,335,273,364,365,379,378,400,377,152,148,176,140,150]

## cheeks
mp_cheeks_R = [127,34,111,101,126,129,165,136,172,58,132]
mp_cheeks_L = [391,358,355,330,340,264,356,454,323,361,288,397,365]

## for extended forehead mp + vgg_heads
extended_indices = [2256,2951,3164,2013,3565,3873,2148,1868,797]
mp_forehead = [151,478,479,480,481,482,483,484,485,486]

mp_ids_dict = {}
mp_ids_dict['mp_lips'] = mp_lips
mp_ids_dict['face_bottom'] = face_bottom
mp_ids_dict['mp_nose'] = mp_nose
mp_ids_dict['mp_eye_Upr_R'] = mp_eye_Upr_R
mp_ids_dict['mp_eye_Upr_L'] = mp_eye_Upr_L
mp_ids_dict['mp_brow_R'] = mp_brow_R
mp_ids_dict['mp_brow_L'] = mp_brow_L
mp_ids_dict['mp_eye_Dwn_R'] = mp_eye_Dwn_R
mp_ids_dict['mp_eye_Dwn_L'] = mp_eye_Dwn_L
mp_ids_dict['mp_lips'] = mp_lips
mp_ids_dict['mp_chin'] = mp_chin
mp_ids_dict['mp_cheeks_R'] = mp_cheeks_R
mp_ids_dict['mp_cheeks_L'] = mp_cheeks_L
mp_ids_dict['extended_indices'] = extended_indices
mp_ids_dict['mp_forehead'] = mp_forehead