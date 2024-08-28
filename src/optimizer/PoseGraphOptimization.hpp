#ifndef __VENOM_SRC_OPTIMIZER_POSE_GRAPH_OPTIMIZATION__
#define __VENOM_SRC_OPTIMIZER_POSE_GRAPH_OPTIMIZATION__

#include <ceres/ceres.h>

//#include "src/manager_env/EnvTrajectory.hpp" 
#include "src/optimizer/factor/PoseGraphSE3Factor.hpp"
#include "src/optimizer/factor/PoseGraphSO3Factor.hpp"
#include "src/utils/IOFuntion.hpp"
#include "src/utils/UtilTransformer.hpp"

namespace simulator {
namespace optimizer {
class PoseGraphOptimization {
public:

  struct RelativePoseCost {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    RelativePoseCost(const Eigen::Quaterniond& q, const Eigen::Vector3d& t)
        : q_(q), t_(t) {}

    template <typename T>
    bool operator()(const T* const q1, const T* const t1,
                  const T* const q2, const T* const t2,
                  T* residuals) const {

      Eigen::Map<const Eigen::Quaternion<T>> Q1(q1);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> T1(t1);
      Eigen::Map<const Eigen::Quaternion<T>> Q2(q2);
      Eigen::Map<const Eigen::Matrix<T, 3, 1>> T2(t2);

      Eigen::Quaternion<T> Q12 = Q1.conjugate() * Q2;
      Eigen::Matrix<T, 3, 1> T12 = Q1.conjugate() * (T2 - T1);

      Eigen::Quaternion<T> Q12_measured(q_.cast<T>());
      Eigen::Quaternion<T> dQ = Q12_measured.conjugate() * Q12;
      residuals[0] = T(2.0) * dQ.x();
      residuals[1] = T(2.0) * dQ.y();
      residuals[2] = T(2.0) * dQ.z();

      Eigen::Matrix<T, 3, 1> T12_measured = t_.cast<T>();
      residuals[3] = T12_measured[0] - T12[0];
      residuals[4] = T12_measured[1] - T12[1];
      residuals[5] = T12_measured[2] - T12[2];

      return true;
    }

    static ceres::CostFunction* Create(const Eigen::Quaterniond& q, const Eigen::Vector3d& t) {
      return (new ceres::AutoDiffCostFunction<RelativePoseCost, 6, 4, 3, 4, 3>(
            new RelativePoseCost(q, t)));
    }

    Eigen::Quaterniond q_;
    Eigen::Vector3d t_;
  };

    static void optimizer(IO::MapOfPoses &poses,
                    IO::VectorOfConstraints &constraints,
                    std::vector<Mat4> &Twcs_posegraph) {

      ceres::Problem problem;

      for (size_t i = 0; i < poses.size(); i++) {
        problem.AddParameterBlock(poses[i].q.coeffs().data(), 4, new ceres::QuaternionParameterization());
        problem.AddParameterBlock(poses[i].p.data(), 3);
      }

      for (size_t i = 0; i < constraints.size(); ++i) {
          ceres::CostFunction* cost_function =
          RelativePoseCost::Create(constraints[i].t_be.q, constraints[i].t_be.p);
          problem.AddResidualBlock(cost_function, nullptr,
                             poses[constraints[i].id_begin].q.coeffs().data(), poses[constraints[i].id_begin].p.data(),
                             poses[constraints[i].id_end].q.coeffs().data(), poses[constraints[i].id_end].p.data());
      }

      ceres::Solver::Options options;
      options.linear_solver_type = ceres::DENSE_QR;
      options.minimizer_progress_to_stdout = true;

      ceres::Solver::Summary summary;
      ceres::Solve(options, &problem, &summary);
      std::cout << summary.FullReport() << std::endl;

      for(int i = 0; i < poses.size(); i++){
        std::cout << "pose "<< i <<", optimized trans: " << poses[i].p.transpose() << std::endl;
      }

      Twcs_posegraph.clear();
      for (const auto& pose : poses) {
          Eigen::Matrix4d Twc = Eigen::Matrix4d::Identity();
          Twc.block<3, 3>(0, 0) = pose.second.q.toRotationMatrix();
          Twc.block<3, 1>(0, 3) = pose.second.p;
          Twcs_posegraph.push_back(Twc);
      }
    }
};


}  // namespace optimizer

}  // namespace simulator
   // problem
#endif  //_VENOM_SRC_OPTIMIZER_POSE_GRAPH_OPTIMIZATION__
