#include "aLibTorch/tquat.h"

namespace a::lt {

Tensor quat_to_tquat(Quat q)
{
    return torch::tensor({q.w(), q.x(), q.y(), q.z()});
}

Quat tquat_to_quat(Tensor q)
{
    return Quat(
        q[0].item<float>(), q[1].item<float>(), q[2].item<float>(), q[3].item<float>()
    );
}

Tensor tquat_to_trot(Tensor quat)
{
    Tensor q0 = quat[0];
    Tensor q1 = quat[1];
    Tensor q2 = quat[2];
    Tensor q3 = quat[3];

    // First row of the rotation matrix
    Tensor r00 = 2 * (q0 * q0 + q1 * q1) - 1;
    Tensor r01 = 2 * (q1 * q2 - q0 * q3);
    Tensor r02 = 2 * (q1 * q3 + q0 * q2);
     
    // Second row of the rotation matrix
    Tensor r10 = 2 * (q1 * q2 + q0 * q3);
    Tensor r11 = 2 * (q0 * q0 + q2 * q2) - 1;
    Tensor r12 = 2 * (q2 * q3 - q0 * q1);
     
    // Third row of the rotation matrix
    Tensor r20 = 2 * (q1 * q3 - q0 * q2);
    Tensor r21 = 2 * (q2 * q3 + q0 * q1);
    Tensor r22 = 2 * (q0 * q0 + q3 * q3) - 1;

    // 3x3 rotation matrix
    Tensor rot_matrix = torch::stack({
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22
    }).view({3, 3});
    return rot_matrix;
}

Tensor trot_to_tqaut(Tensor rot)
{
    rot = rot.conj().transpose(0, 1);

    float r00 = rot[0][0].item<float>();
    float r11 = rot[1][1].item<float>();
    float r22 = rot[2][2].item<float>();
    
    if(r22 < 0.0f)
    {
        if(r00 > r11)
        {
            Tensor t = 1.0f + rot[0][0] - rot[1][1] - rot[2][2];
            Tensor q = torch::stack({
                rot[1][2] - rot[2][1],
                t,
                rot[0][1] + rot[1][0],
                rot[2][0] + rot[0][2]
            });
            q = q * 0.5f / t.sqrt();
            return q;
        }
        else
        {
            Tensor t = 1.0f - rot[0][0] + rot[1][1] - rot[2][2];
            Tensor q = torch::stack({
                rot[2][0] - rot[0][2],
                rot[0][1] + rot[1][0],
                t,
                rot[1][2] + rot[2][1]
            });
            q = q * 0.5f / t.sqrt();
            return q;
        }
    }
    else
    {
        if(r00 < -r11)
        {
            Tensor t = 1.0f - rot[0][0] - rot[1][1] + rot[2][2];
            Tensor q = torch::stack({
                rot[0][1] - rot[1][0],
                rot[2][0] + rot[0][2],
                rot[1][2] + rot[2][1],
                t
            });
            q = q * 0.5f / t.sqrt();
            return q;
        }
        else
        {
            Tensor t = 1.0f + rot[0][0] + rot[1][1] + rot[2][2];
            Tensor q = torch::stack({
                t,
                rot[1][2] - rot[2][1],
                rot[2][0] - rot[0][2],
                rot[0][1] - rot[1][0]
            });
            q = q * 0.5f / t.sqrt();
            return q;
        }
    }
}

Tensor tquat_mul(Tensor q2, Tensor q1)
{
    q1 = q1.view({4, 1});
    q2 = q2.view({1, 4});
    Tensor terms = torch::mm(q1, q2);

    Tensor w = terms[0][0] - terms[1][1] - terms[2][2] - terms[3][3];
    Tensor x = terms[0][1] + terms[1][0] - terms[2][3] + terms[3][2];
    Tensor y = terms[0][2] + terms[1][3] + terms[2][0] - terms[3][1];
    Tensor z = terms[0][3] - terms[1][2] + terms[2][1] + terms[3][0];

    Tensor result = torch::stack({w, x, y, z});
    return result;
}

Tensor tquat_rot(Tensor q, Tensor v)
{
    Tensor qvec = q.narrow(0, 1, 3);
    Tensor uv = torch::cross(qvec, v, 0);
    Tensor uuv = torch::cross(qvec, uv, 0);
    return (v + 2.0f * (q[0] * uv + uuv));
}

Tensor tquat_to_taaxis(Tensor q)
{
#if 0
    float tolerance = 1e-7;

    Tensor vec = q.narrow(0, 1, 3);
    Tensor v_norm = vec.norm();

    if(v_norm.item<float>() > tolerance)
    {
        vec = vec / v_norm;
    }
    
    Tensor magnitude = torch::exp(q[0]);

    Tensor q_scalar = magnitude * torch::cos(v_norm);
    Tensor q_vector = magnitude * torch::sin(v_norm) * vec;
    
    std::cout << "q_scalar\n" << q_scalar << std::endl;
    std::cout << "q_vector\n" << q_vector << std::endl;
    Tensor q_exp = torch::stack({q_scalar, q_vector[0], q_vector[1], q_vector[2]});
    return q_exp;
#else
    Tensor vec = q.narrow(0, 1, 3);
    if(vec.norm().item<float>() > 1e-7)
    {
        Tensor angle = 2.0f * torch::atan2(vec.norm(), q[0]);
        Tensor axis = vec / vec.norm();
        return angle * axis;
    }
    else
    {
        return 0.0f * vec;
    }
#endif
}

Tensor taaxis_to_tquat(Tensor aaxis)
{
    Tensor angle = aaxis.norm();
    if(angle.item<float>() > 1e-7)
    {
        Tensor axis = aaxis / aaxis.norm();
        Tensor theta = angle / 2.0f;
        Tensor r = torch::cos(theta);
        Tensor i = axis * torch::sin(theta);
        return torch::stack({r, i[0], i[1], i[2]});
    }
    else
    {
        Tensor axis = 0.0f * aaxis;
        Tensor theta = 0.0f * angle;
        Tensor r = torch::cos(theta); // 1.0
        Tensor i = axis * torch::sin(theta); // zero vec
        return torch::stack({r, i[0], i[1], i[2]});
    }
}

}
