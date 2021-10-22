#include "torchtype.h"

namespace a::lt {

// Reference
//   https://github.com/facebookresearch/QuaterNet/blob/master/common/quaternion.py
//   https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py

/**
 * @return      Quaternion tensor {w, x, y, z}
 */
Tensor quat_to_tquat(Quat q);

/**
 * @param q     quaternion tensor {w, x, y, z}
 * @return      quaternion
 */
Quat tquat_to_quat(Tensor q);

/**
 * @brief       Covert a quaternion into a full three-dimensional rotation matrix.
 *              Calling this method will implicitly normalise the Quaternion.
 *              ref: https://automaticaddison.com/how-to-convert-a-quaternion-to-a-rotation-matrix/
 * @param quat  quaternion tensor: [4]
 * @return      rotation matrix tensor: [3, 3]
 */
Tensor tquat_to_trot(Tensor quat);

/**
 * @brief     This code uses a modification of the algorithm described in:
 *            https://d3cw3dd2w32x2b.cloudfront.net/wp-content/uploads/2015/01/matrix-to-quat.pdf
 *            which is itself based on the method described here:
 *            http://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/
 *            ref: https://github.com/KieranWynn/pyquaternion/blob/master/pyquaternion/quaternion.py
 * @param rot rotation matrix tensor: [3, 3]
 */
Tensor trot_to_tqaut(Tensor rot);

/**
 * @brief    Multiply quaternion q2 with quaternion q1
 *           ref: https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py#L13
 * @return   q2 * q1
 */
Tensor tquat_mul(Tensor q2, Tensor q1);

/**
 * @brief   Rotate vector v about the rotation described by quaternion q.
 *          ref: https://github.com/facebookresearch/QuaterNet/blob/9d8485b732b0a44b99b6cf4b12d3915703507ddc/common/quaternion.py#L33
 * @param q quat tensor: [4]
 * @param v vec3 tensor: [3]
 * @return  q.rotate(v)
 */
Tensor tquat_rot(Tensor q, Tensor v);

/**
 * @brief Convert quaternion to axis-angle rotation
 */
Tensor tquat_to_taaxis(Tensor q);

/**
 * @brief Convert axis-angle rotation to quaternion
 */
Tensor taaxis_to_tquat(Tensor aaxis);
}
