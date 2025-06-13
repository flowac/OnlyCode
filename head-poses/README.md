## Head pose and eye tracking

1. Get data from FaceLandmarker
2. Determine baseline head and eye positions (calibrate in beginning?)
3. Determine if the user is periodically checking another screen for answers

## Detections

For example these values are within 0 -> 1 where 1 is the maximum value.
Keywords:

- out -> looking away from nose
- in -> towards the nose

1. Example of looking
   | Feature | High Value | Meaning |
   | ----------------- | ---------- | ---------------------------- |
   | `eyeLookOutRight` | > 0.3 | Right eye is looking outward |
   | `eyeLookInLeft` | > 0.3 | Left eye is turned inward |

2. Example of looking
   | Feature | High Value | Meaning |
   | ---------------- | ---------- | --------------------------- |
   | `eyeLookOutLeft` | > 0.3 | Left eye is looking outward |
   | `eyeLookInRight` | > 0.3 | Right eye is turned inward |

3. General

   | Looking Direction | Eye   | Rotation           | Blendshape        | Expected Value |
   | ----------------- | ----- | ------------------ | ----------------- | -------------- |
   | Left              | Right | Outward (temporal) | `eyeLookOutRight` | ↑ (high)       |
   |                   | Left  | Inward (nasal)     | `eyeLookInLeft`   | ↑ (high)       |
   | Right             | Left  | Outward            | `eyeLookOutLeft`  | ↑ (high)       |
   |                   | Right | Inward             | `eyeLookInRight`  | ↑ (high)       |

### Head Pose Angles

| Angle   | Value (°) | Interpretation                                                                                   |
| ------- | --------- | ------------------------------------------------------------------------------------------------ |
| `yaw`   | 5.48      | Head is turned **slightly right**.                                                               |
| `pitch` | 16.89     | Head is tilted **upward**, like the person is **looking above the camera**.                      |
| `roll`  | 6.56      | Head is tilted slightly to the **right side**, like a slight **lean toward the right shoulder**. |

| Value (degrees) | Interpretation                   |
| --------------- | -------------------------------- |
| `0`             | Facing **straight forward**      |
| Positive (`+`)  | Turned to the person’s **right** |
| Negative (`–`)  | Turned to the person’s **left**  |

1. Yaw Values
   ±10–15°: Slight turn

   ±30°: Noticeable turn (e.g., looking at a second monitor)

   ±45°+: Head turned significantly, probably not paying attention to the screen

2. Pitch Thresholds:

   +15–30°: Looking noticeably above the screen

   –15–30°: Looking down (e.g., at a phone or desk)

   –45°+: Looking down sharply — possibly inattentive or reading

3. Roll
   | Value (degrees) | Interpretation |
   | --------------- | ------------------------------------ |
   | `0` | Head is **upright** |
   | Positive (`+`) | Tilted toward the **right shoulder** |
   | Negative (`–`) | Tilted toward the **left shoulder** |

Leverage the nlohmann c++ library to find if either of these categories exist / get their values
