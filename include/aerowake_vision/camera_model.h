#pragma once

#include <vector>
#include <Eigen/Core>

using namespace std;
using namespace Eigen;


class Camera
{
public:
    typedef Matrix<double,2,1> Vec2;
    typedef Matrix<double,2,2> Mat2;
    typedef Matrix<double,3,1> Vec3;
    typedef Matrix<double,5,1> Vec5;

    Vec2 focal_len_;
    Vec2 cam_center_;
    Vec5 distortion_;
    Vector2d image_size_;

    Camera() : focal_len_(0.0, 0.0), cam_center_(0.0, 0.0), distortion_(Vec5::Zero()) {}

    Camera(const Vec2& f, const Vec2& c, const Vec5& d) :
        focal_len_(f),
        cam_center_(c),
        distortion_(d)
    {}

    Camera(const Vec2& f, const Vec2& c, const Vec5& d, const Vector2d& size) :
        focal_len_(f.data()),
        cam_center_(c.data()),
        distortion_(d.data())
    {
        setSize(size);
    }

    Camera(const Camera &copy)
    {
        focal_len_ = copy.focal_len_;
        cam_center_ = copy.cam_center_;
        distortion_ = copy.distortion_;
    }

    Camera& operator=(const Camera& other)
    {
        focal_len_ = other.focal_len_;
        cam_center_ = other.cam_center_;
        distortion_ = other.distortion_;
        return *this;
    }

    void setSize(const Vector2d& size)
    {
        image_size_ = size;
    }

    void unDistort(const Vec2& pi_u, Vec2& pi_d) const
    {
        const double k1 = distortion_(0);
        const double k2 = distortion_(1);
        const double p1 = distortion_(2);
        const double p2 = distortion_(3);
        const double k3 = distortion_(4);
        const double x = pi_u.x();
        const double y = pi_u.y();
        const double xy = x*y;
        const double xx = x*x;
        const double yy = y*y;
        const double rr = xx*yy;
        const double r4 = rr*rr;
        const double r6 = r4*rr;


        // https://docs.opencv.org/2.4/doc/tutorials/calib3d/camera_calibration/camera_calibration.html
        const double g =  1.0 + k1 * rr + k2 * r4 + k3*r6;
        const double dx = 2.0 * p1 * xy + p2 * (rr + 2.0 * xx);
        const double dy = 2.0 * p2 * xy + p1 * (rr + 2.0 * yy);

        pi_d.x() = g * (x + dx);
        pi_d.y() = g * (y + dy);
    }

    void Distort(const Vec2& pi_d, Vec2& pi_u, double tol=1e-6) const
    {
        pi_u = pi_d;
        Vec2 pihat_d;
        Mat2 J;
        Vec2 e;
        double prev_e = 1000.0;
        double enorm = 0.0;

        static const int max_iter = 50;
        int i = 0;
        while (i < max_iter)
        {
            unDistort(pi_u, pihat_d);
            e = pihat_d - pi_d;
            enorm = e.norm();
            if (enorm <= tol || prev_e < enorm)
                break;
            prev_e = enorm;

            distortJac(pi_u, J);
            pi_u = pi_u - J*e;
            i++;
        }
//        if ((pi_u.array() != pi_u.array()).any())
//        {
//            int debug = 1;
//        }
    }

    void distortJac(const Vec2& pi_u, Mat2& J) const
    {
        const double k1 = distortion_(0);
        const double k2 = distortion_(1);
        const double p1 = distortion_(2);
        const double p2 = distortion_(3);
        const double k3 = distortion_(4);

        const double x = pi_u.x();
        const double y = pi_u.y();
        const double xy = x*y;
        const double xx = x*x;
        const double yy = y*y;
        const double rr = xx+yy;
        const double r = sqrt(rr);
        const double r4 = rr*rr;
        const double r6 = rr*r4;
        const double g =  1.0 + k1 * rr + k2 * r4 + k3*r6;
        const double dx = (x + (2.0*p1*xy + p2*(rr+2.0*xx)));
        const double dy = (y + (p1*(rr+2.0*yy) + 2.0*p2*xy));

        const double drdx = x / r;
        const double drdy = y / r;
        const double dgdx = k1*2.0*r*drdx + 4.0*k2*rr*r*drdx + 6.0*k3*r4*r*drdx;
        const double dgdy = k1*2.0*r*drdy + 4.0*k2*rr*r*drdy + 6.0*k3*r4*r*drdy;

        J << /* dxbar/dx */ (1.0 + (2.0*p1*y + p2*(2.0*r*drdx + 4.0*x)))*g + dx*dgdx,
             /* dxbar/dy */ (2.0*p1*x + p2*2.0*r*drdy)*g + dx*dgdy,
             /* dybar/dx */ (p1*2.0*r*drdx+2.0*p2*y)*g + dy*dgdx,
             /* dybar/dy */ (1.0 + (p1*(2.0*r*drdy + 4.0*y) + 2.0*p2*x))*g + dy*dgdy;

//        if ((J.array() != J.array()).any())
//        {
//            int debug = 1;
//        }
    }


    void pix2intrinsic(const Vec2& pix, Vec2& pi) const
    {
        const double fx = focal_len_.x();
        const double fy = focal_len_.y();
        const double cx = cam_center_.x();
        const double cy = cam_center_.y();
        // ZERO SKEW
        pi << (1.0/fx) * (pix.x() - cx),
              (1.0/fy) * (pix.y() - cy);
    }

    void intrinsic2pix(const Vec2& pi, Vec2& pix) const
    {
        const double fx = focal_len_.x();
        const double fy = focal_len_.y();
        const double cx = cam_center_.x();
        const double cy = cam_center_.y();
        // ZERO SKEW
        pix << fx*pi.x() + cx,
               fy*pi.y() + cy;
    }

    void proj(const Vec3& pt, Vec2& pix) const // TRUNCATING FOR PURPOSES OF THE AEROWAKE SIM
    {
        const double pt_z = pt(2);
        Vec2 pi_d;
        if (pt_z > 0)
        {
            Vec2 pi_u = (pt.segment<2>(0) / pt_z);
            Distort(pi_u, pi_d);
            intrinsic2pix(pi_d, pix);
//            intrinsic2pix(pi_u, pix);
        }
        else
            pix = Vec2(-9e10,-9e10);
    }

    inline bool check(const Vector2d& pix) const
    {
        return !((pix.array() > image_size_.array()).any()|| (pix.array() < 0).any());
    }

    void invProj(const Vec2& pix, const double& depth, Vec3& pt) const
    {
        Vec2 pi_d, pi_u;
        pix2intrinsic(pix, pi_d);
        unDistort(pi_d, pi_u);
        pt.segment<2>(0) = pi_u;
        pt(2) = 1.0;
        pt *= depth / pt.norm();
    }

private:
    const Matrix2d I_2x2 = Matrix2d::Identity();
};


