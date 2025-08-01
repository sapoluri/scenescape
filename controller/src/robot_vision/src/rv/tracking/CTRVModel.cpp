// SPDX-FileCopyrightText: 2019 - 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "rv/tracking/CTRVModel.hpp"

namespace rv {
namespace tracking {

void CTRVModel::stateConversionFunction(const cv::Mat &x_k, const cv::Mat &u_k, const cv::Mat &v_k, cv::Mat &x_kplus1)
{
  cv::Mat vk = v_k.clone();

  /*
   * The time is considered the control input
   */
  double deltaT = u_k.at<double>(0, 0);

  double x = x_k.at<double>(0, 0);
  double y = x_k.at<double>(1, 0);
  double vx = x_k.at<double>(2, 0);
  double vy = x_k.at<double>(3, 0);
  double ax = x_k.at<double>(4, 0);
  double ay = x_k.at<double>(5, 0);
  double w = x_k.at<double>(11, 0);
  double yaw = x_k.at<double>(10, 0);

  double v = sqrt(vx * vx + vy * vy);
  double velocityYaw = atan2(vy, vx);     // Velocity yaw

  double nextYaw = velocityYaw + w * deltaT; // Velocity yaw for the next time iteration

  double cosNextYaw = cos(nextYaw);
  double sinNextYaw = sin(nextYaw);


  if (fabs(w) >= 1e-3)
  {
    x_kplus1.at<double>(0, 0) = x + v / w * (sinNextYaw - sin(velocityYaw)); // Position in X
    x_kplus1.at<double>(1, 0) = y + v / w * (cos(velocityYaw) - cosNextYaw); // Position in Y
  }
  else
  {
    // as the turn rate converges to zero, the equations are:
    x_kplus1.at<double>(0, 0) = x + v * cosNextYaw * deltaT; // Position in X
    x_kplus1.at<double>(1, 0) = y + v * sinNextYaw * deltaT; // Position in Y
  }

  x_kplus1.at<double>(2, 0) = v * cosNextYaw;            // Velocity in X
  x_kplus1.at<double>(3, 0) = v * sinNextYaw;            // Velocity in Y
  x_kplus1.at<double>(4, 0) = 0.;                     // Acceleration in X
  x_kplus1.at<double>(5, 0) = 0.;                     // Acceleration in Y
  x_kplus1.at<double>(6, 0) = x_k.at<double>(6, 0);   // Position in Z
  x_kplus1.at<double>(7, 0) = x_k.at<double>(7, 0);   // Length
  x_kplus1.at<double>(8, 0) = x_k.at<double>(8, 0);   // Width
  x_kplus1.at<double>(9, 0) = x_k.at<double>(9, 0);   // Height
  x_kplus1.at<double>(10, 0) = x_k.at<double>(10, 0); // Yaw
  x_kplus1.at<double>(11, 0) = w;                     // Yaw Rate

  x_kplus1 += vk; // additive process noise
}

void CTRVModel::measurementFunction(const cv::Mat &x_k, const cv::Mat &n_k, cv::Mat &z_k)
{
  z_k.at<double>(0, 0) = x_k.at<double>(0, 0);  // Position in X
  z_k.at<double>(1, 0) = x_k.at<double>(1, 0);  // Position in Y
  z_k.at<double>(2, 0) = x_k.at<double>(6, 0);  // Position in Z
  z_k.at<double>(3, 0) = x_k.at<double>(7, 0);  // Length
  z_k.at<double>(4, 0) = x_k.at<double>(8, 0);  // Width
  z_k.at<double>(5, 0) = x_k.at<double>(9, 0);  // Height
  z_k.at<double>(6, 0) = x_k.at<double>(10, 0); // Yaw
  z_k += n_k;                                   // additive measurement noise
}
} // namespace tracking
} // namespace rv
