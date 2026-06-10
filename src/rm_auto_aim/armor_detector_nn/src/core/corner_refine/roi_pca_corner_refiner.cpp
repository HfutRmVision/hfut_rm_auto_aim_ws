#include "armor_detector_nn/core/corner_refine/roi_pca_corner_refiner.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <vector>

#include <opencv2/imgproc.hpp>

#include <rm_utils/logger/log.hpp>

namespace fyt::auto_aim {

namespace {

double elapsedMs(
    const std::chrono::steady_clock::time_point& start) {
  return std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - start).count();
}

float sampleGrayBilinear(const cv::Mat& gray, float x, float y) {
  if (x < 0.0f || y < 0.0f ||
      x >= static_cast<float>(gray.cols - 1) ||
      y >= static_cast<float>(gray.rows - 1)) {
    return 0.0f;
  }

  const int x0 = static_cast<int>(std::floor(x));
  const int y0 = static_cast<int>(std::floor(y));
  const float ax = x - static_cast<float>(x0);
  const float ay = y - static_cast<float>(y0);

  const float v00 = static_cast<float>(gray.at<uchar>(y0, x0));
  const float v10 = static_cast<float>(gray.at<uchar>(y0, x0 + 1));
  const float v01 = static_cast<float>(gray.at<uchar>(y0 + 1, x0));
  const float v11 = static_cast<float>(gray.at<uchar>(y0 + 1, x0 + 1));

  const float v0 = v00 * (1.0f - ax) + v10 * ax;
  const float v1 = v01 * (1.0f - ax) + v11 * ax;
  return v0 * (1.0f - ay) + v1 * ay;
}

bool isFinitePoint(const cv::Point2f& p) {
  return std::isfinite(p.x) && std::isfinite(p.y);
}

double safeLogRatio(double a, double b) {
  return std::log(std::max(1e-3, a) / std::max(1e-3, b));
}

double polygonArea(const std::array<cv::Point2f, 4>& pts) {
  return std::abs(cv::contourArea(
    std::vector<cv::Point2f>(pts.begin(), pts.end())));
}

double edgeAngle(const cv::Point2f& a, const cv::Point2f& b) {
  const cv::Point2f d = b - a;
  return std::atan2(d.y, d.x);
}

struct QuadShape {
  bool valid{false};
  double area{0.0};
  double aspect{0.0};
  double side_ratio{0.0};
  double left_right_log_ratio{0.0};
  double top_bottom_log_ratio{0.0};
  std::array<double, 4> edge_angles{};
};

QuadShape computeQuadShape(const std::array<cv::Point2f, 4>& corners) {
  QuadShape shape;
  for (const auto& corner : corners) {
    if (!isFinitePoint(corner)) {
      return shape;
    }
  }

  const double left = cv::norm(corners[0] - corners[1]);
  const double top = cv::norm(corners[1] - corners[2]);
  const double right = cv::norm(corners[3] - corners[2]);
  const double bottom = cv::norm(corners[0] - corners[3]);
  if (left < 1e-3 || right < 1e-3 || top < 1e-3 || bottom < 1e-3) {
    return shape;
  }

  const double avg_width = (top + bottom) * 0.5;
  const double avg_height = (left + right) * 0.5;
  shape.valid = true;
  shape.area = polygonArea(corners);
  shape.aspect = avg_width / std::max(1e-3, avg_height);
  shape.side_ratio = std::max(left, right) / std::min(left, right);
  shape.left_right_log_ratio = safeLogRatio(left, right);
  shape.top_bottom_log_ratio = safeLogRatio(top, bottom);
  shape.edge_angles = {{
    edgeAngle(corners[0], corners[1]),
    edgeAngle(corners[1], corners[2]),
    edgeAngle(corners[3], corners[2]),
    edgeAngle(corners[0], corners[3])
  }};
  return shape;
}

}  // namespace

RoiPcaCornerRefiner::RoiPcaCornerRefiner(
    const CornerRefineConfig& config,
    int frame_cols,
    int frame_rows)
  : config_(config),
    frame_cols_(frame_cols),
    frame_rows_(frame_rows)
{
}

double RoiPcaCornerRefiner::angleDistance(double a, double b) {
  double d = a - b;
  while (d > CV_PI / 2.0) d -= CV_PI;
  while (d < -CV_PI / 2.0) d += CV_PI;
  return std::abs(d);
}

cv::Rect RoiPcaCornerRefiner::extractArmorROI(
    const std::array<cv::Point2f, 4>& corners) const
{
  for (const auto& corner : corners) {
    if (!isFinitePoint(corner)) {
      return {};
    }
  }

  // Canonical order in this package:
  // [0]=left_bottom, [1]=left_top, [2]=right_top, [3]=right_bottom.
  // SP25 expands the NN quadrilateral along the two light-bar axes first,
  // then expands horizontally.  This keeps the ROI aligned with the armor
  // projection, which matters when a yawed armor becomes a trapezoid.
  const float scale = static_cast<float>(
    std::max(1.0, config_.full_roi_expand_ratio));

  const cv::Point2f left_center = (corners[0] + corners[1]) * 0.5f;
  const cv::Point2f right_center = (corners[2] + corners[3]) * 0.5f;
  const cv::Point2f left_axis = corners[0] - corners[1];
  const cv::Point2f right_axis = corners[3] - corners[2];

  if (cv::norm(left_axis) < 1.0f || cv::norm(right_axis) < 1.0f ||
      cv::norm(right_center - left_center) < 1.0f) {
    return {};
  }

  const cv::Point2f lt_ext = left_center - left_axis * (0.5f * scale);
  const cv::Point2f lb_ext = left_center + left_axis * (0.5f * scale);
  const cv::Point2f rt_ext = right_center - right_axis * (0.5f * scale);
  const cv::Point2f rb_ext = right_center + right_axis * (0.5f * scale);

  const cv::Point2f top_mid = (lt_ext + rt_ext) * 0.5f;
  const cv::Point2f bottom_mid = (lb_ext + rb_ext) * 0.5f;
  const cv::Point2f top_half = (rt_ext - lt_ext) * (0.5f * scale);
  const cv::Point2f bottom_half = (rb_ext - lb_ext) * (0.5f * scale);

  const std::vector<cv::Point2f> pts = {
    top_mid - top_half,
    top_mid + top_half,
    bottom_mid + bottom_half,
    bottom_mid - bottom_half,
  };
  cv::Rect2f box = cv::boundingRect(pts);
  if (box.width <= 1.0f || box.height <= 1.0f) {
    return {};
  }

  const int x0 = std::max(0, static_cast<int>(std::floor(box.x)));
  const int y0 = std::max(0, static_cast<int>(std::floor(box.y)));
  const int x1 = std::min(frame_cols_, static_cast<int>(std::ceil(box.x + box.width)));
  const int y1 = std::min(frame_rows_, static_cast<int>(std::ceil(box.y + box.height)));
  if (x1 <= x0 || y1 <= y0) {
    return {};
  }
  return cv::Rect(cv::Point(x0, y0), cv::Point(x1, y1));
}

bool RoiPcaCornerRefiner::acceptLightBar(
    const LightBarCandidate& lightbar) const
{
  if (lightbar.area < config_.min_contour_area_px) {
    return false;
  }
  if (lightbar.length < config_.min_lightbar_length_px) {
    return false;
  }
  if (lightbar.ratio < config_.min_lightbar_ratio ||
      lightbar.ratio > config_.max_lightbar_ratio) {
    return false;
  }

  const double max_angle =
    std::max(0.0, config_.max_lightbar_angle_error_deg) * CV_PI / 180.0;
  return angleDistance(lightbar.angle, CV_PI / 2.0) <= max_angle;
}

std::array<cv::Point2f, 4> RoiPcaCornerRefiner::cornersFromLightBars(
    const LightBarCandidate& left,
    const LightBarCandidate& right)
{
  return {{
    left.bottom,
    left.top,
    right.top,
    right.bottom
  }};
}

std::vector<RoiPcaCornerRefiner::LightBarCandidate>
RoiPcaCornerRefiner::detectLightBarsInRoi(
    const cv::Mat& frame,
    const cv::Rect& roi,
    fyt::EnemyColor color) const
{
  std::vector<LightBarCandidate> lightbars;
  if (roi.width < 3 || roi.height < 3) {
    return lightbars;
  }

  cv::Mat roi_img = frame(roi);
  cv::Mat enhanced;
  if (config_.method == "sp25_lightbar" ||
      config_.method == "full_lightbar_roi" ||
      color == fyt::EnemyColor::WHITE) {
    cv::cvtColor(roi_img, enhanced, cv::COLOR_BGR2GRAY);
  } else {
    std::vector<cv::Mat> channels;
    cv::split(roi_img, channels);
    if (color == fyt::EnemyColor::RED) {
      cv::subtract(channels[2], channels[0], enhanced);
    } else {
      cv::subtract(channels[0], channels[2], enhanced);
    }
  }

  cv::Mat binary;
  if (config_.binary_threshold > 0.0) {
    cv::threshold(enhanced, binary, config_.binary_threshold, 255, cv::THRESH_BINARY);
  } else {
    cv::threshold(enhanced, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  }

  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
  cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);

  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

  const cv::Point2f offset(static_cast<float>(roi.x), static_cast<float>(roi.y));
  for (const auto& contour : contours) {
    const double area = cv::contourArea(contour);
    if (area < config_.min_contour_area_px) {
      continue;
    }

    cv::RotatedRect rect = cv::minAreaRect(contour);
    if (rect.size.width <= 1.0f || rect.size.height <= 1.0f) {
      continue;
    }

    std::array<cv::Point2f, 4> corners;
    rect.points(corners.data());
    std::sort(corners.begin(), corners.end(),
              [](const cv::Point2f& a, const cv::Point2f& b) {
                if (std::abs(a.y - b.y) > 1e-3f) {
                  return a.y < b.y;
                }
                return a.x < b.x;
              });

    cv::Point2f top = (corners[0] + corners[1]) * 0.5f + offset;
    cv::Point2f bottom = (corners[2] + corners[3]) * 0.5f + offset;
    cv::Point2f axis = bottom - top;
    double length = cv::norm(axis);
    double width = cv::norm(corners[0] - corners[1]);
    if (width < 1e-3) {
      width = std::min(rect.size.width, rect.size.height);
    }
    if (width < 1e-3) {
      continue;
    }

    LightBarCandidate lightbar;
    lightbar.top = top;
    lightbar.bottom = bottom;
    lightbar.center = (top + bottom) * 0.5f;
    lightbar.angle = std::atan2(axis.y, axis.x);
    lightbar.length = length;
    lightbar.width = width;
    lightbar.ratio = length / width;
    lightbar.area = area;
    if (acceptLightBar(lightbar)) {
      lightbars.push_back(lightbar);
    }
  }

  std::sort(lightbars.begin(), lightbars.end(),
            [](const LightBarCandidate& a, const LightBarCandidate& b) {
              return a.center.x < b.center.x;
            });
  return lightbars;
}

double RoiPcaCornerRefiner::pairScore(
    const LightBarCandidate& left,
    const LightBarCandidate& right,
    const ArmorDetection& detection,
    const std::array<cv::Point2f, 4>& refined) const
{
  if (left.center.x >= right.center.x) {
    return std::numeric_limits<double>::infinity();
  }

  const double side_ratio =
    std::max(left.length, right.length) /
    std::max(1e-3, std::min(left.length, right.length));
  if (side_ratio > config_.max_side_ratio) {
    return std::numeric_limits<double>::infinity();
  }

  const cv::Point2f left_to_right = right.center - left.center;
  const double roll = std::atan2(left_to_right.y, left_to_right.x);
  const double left_rect_error = angleDistance(left.angle, roll + CV_PI / 2.0);
  const double right_rect_error = angleDistance(right.angle, roll + CV_PI / 2.0);
  const double rect_error = std::max(left_rect_error, right_rect_error);
  const double max_rect_error =
    std::max(0.0, config_.max_rectangular_error_deg) * CV_PI / 180.0;
  if (config_.max_rectangular_error_deg > 0.0 &&
      rect_error > max_rect_error) {
    return std::numeric_limits<double>::infinity();
  }

  const double left_match_error =
    cv::norm(left.bottom - detection.keypoints[0]) +
    cv::norm(left.top - detection.keypoints[1]);
  const double right_match_error =
    cv::norm(right.top - detection.keypoints[2]) +
    cv::norm(right.bottom - detection.keypoints[3]);
  const double match_error =
    (left_match_error + right_match_error) * 0.25;
  const double worst_lightbar_error =
    std::max(left_match_error, right_match_error) * 0.5;
  if (match_error > config_.max_lightbar_match_error_px) {
    return std::numeric_limits<double>::infinity();
  }
  if (worst_lightbar_error > config_.max_lightbar_match_error_px) {
    return std::numeric_limits<double>::infinity();
  }

  const cv::Point2f refined_center =
    (refined[0] + refined[1] + refined[2] + refined[3]) * 0.25f;
  const double center_shift = cv::norm(refined_center - detection.center);
  if (center_shift > config_.max_pair_center_shift_px) {
    return std::numeric_limits<double>::infinity();
  }

  double max_shift = 0.0;
  for (int k = 0; k < 4; ++k) {
    max_shift = std::max(
      max_shift, static_cast<double>(cv::norm(refined[k] - detection.keypoints[k])));
  }
  if (config_.max_corner_shift_px > 0.0 &&
      max_shift > config_.max_corner_shift_px) {
    return std::numeric_limits<double>::infinity();
  }

  return match_error + 0.35 * worst_lightbar_error + 0.25 * center_shift +
         10.0 * rect_error + 2.0 * std::abs(side_ratio - 1.0);
}

RefineResult RoiPcaCornerRefiner::refineByLightbarRoi(
    const cv::Mat& frame,
    const ArmorDetection& detection,
    const std::chrono::steady_clock::time_point& start)
{
  RefineResult result;
  result.refined_keypoints = detection.keypoints;

  const cv::Rect roi = extractArmorROI(detection.keypoints);
  if (roi.width < 3 || roi.height < 3 || roi.area() < 20) {
    result.reason = RefineFailReason::ROI_TOO_SMALL;
    result.elapsed_ms = elapsedMs(start);
    return result;
  }

  if (roi.area() > frame.cols * frame.rows) {
    result.reason = RefineFailReason::ROI_TOO_LARGE;
    result.elapsed_ms = elapsedMs(start);
    return result;
  }

  if (elapsedMs(start) > config_.time_budget_ms) {
    result.reason = RefineFailReason::TIMEOUT;
    result.elapsed_ms = elapsedMs(start);
    return result;
  }

  const auto lightbars = detectLightBarsInRoi(frame, roi, detection.color);
  if (lightbars.size() < 2) {
    result.reason = RefineFailReason::NO_LIGHTBAR_PAIR;
    result.elapsed_ms = elapsedMs(start);
    return result;
  }

  double best_score = std::numeric_limits<double>::infinity();
  std::array<cv::Point2f, 4> best = detection.keypoints;
  for (size_t i = 0; i < lightbars.size(); ++i) {
    for (size_t j = i + 1; j < lightbars.size(); ++j) {
      const auto& left = lightbars[i];
      const auto& right = lightbars[j];
      if (left.center.x >= right.center.x) {
        continue;
      }

      const auto refined = cornersFromLightBars(left, right);
      if (!validateGeometry(refined, detection.keypoints)) {
        continue;
      }

      const double score = pairScore(left, right, detection, refined);
      if (score < best_score) {
        best_score = score;
        best = refined;
      }
    }
  }

  if (!std::isfinite(best_score)) {
    result.reason = RefineFailReason::NO_LIGHTBAR_PAIR;
    result.elapsed_ms = elapsedMs(start);
    return result;
  }

  result.refine_quality = computeQuality(best, detection.keypoints);
  if (config_.min_refine_quality > 0.0 &&
      result.refine_quality < config_.min_refine_quality) {
    result.reason = RefineFailReason::REFINE_WORSE;
    result.elapsed_ms = elapsedMs(start);
    return result;
  }

  result.ok = true;
  result.refined_keypoints = best;
  result.elapsed_ms = elapsedMs(start);
  result.reason = RefineFailReason::NONE;
  return result;
}

std::pair<cv::Rect2f, cv::Rect2f>
RoiPcaCornerRefiner::extractLightBarROIs(
    const std::array<cv::Point2f, 4>& corners)
{
  // Canonical order:
  // [0]=left_bottom, [1]=left_top, [2]=right_top, [3]=right_bottom.
  // Left light bar uses [0] and [1].
  cv::Point2f lc = (corners[0] + corners[1]) * 0.5f;
  float lh_half = cv::norm(corners[0] - corners[1]) * 0.5f * config_.roi_expand_ratio;
  float lw_half = lh_half * 0.4f;

  cv::Rect2f left_roi(
    lc.x - lw_half, lc.y - lh_half,
    lw_half * 2.0f, lh_half * 2.0f);

  // Right light bar uses [2] and [3].
  cv::Point2f rc = (corners[2] + corners[3]) * 0.5f;
  float rh_half = cv::norm(corners[2] - corners[3]) * 0.5f * config_.roi_expand_ratio;
  float rw_half = rh_half * 0.4f;

  cv::Rect2f right_roi(
    rc.x - rw_half, rc.y - rh_half,
    rw_half * 2.0f, rh_half * 2.0f);

  left_roi  &= cv::Rect2f(0, 0, static_cast<float>(frame_cols_),
                           static_cast<float>(frame_rows_));
  right_roi &= cv::Rect2f(0, 0, static_cast<float>(frame_cols_),
                           static_cast<float>(frame_rows_));

  return {left_roi, right_roi};
}

RoiPcaCornerRefiner::LightBarEndpoints
RoiPcaCornerRefiner::refineLightBar(
    const cv::Mat& frame,
    const cv::Rect2f& roi,
    fyt::EnemyColor color)
{
  LightBarEndpoints result;

  cv::Rect roi_int(
    static_cast<int>(roi.x), static_cast<int>(roi.y),
    static_cast<int>(roi.width), static_cast<int>(roi.height));
  roi_int &= cv::Rect(0, 0, frame.cols, frame.rows);
  if (roi_int.width < 3 || roi_int.height < 3) {
    result.reason = RefineFailReason::ROI_TOO_SMALL;
    return result;
  }

  cv::Mat roi_img = frame(roi_int);

  // Channel enhancement: enhance the target color.
  cv::Mat gray;
  {
    std::vector<cv::Mat> channels;
    cv::split(roi_img, channels);
    // BGR order: channels[0]=B, channels[1]=G, channels[2]=R
    if (color == fyt::EnemyColor::RED) {
      gray = channels[2] - channels[0];  // R - B
    } else {
      gray = channels[0] - channels[2];  // B - R
    }
  }

  // Binarization. Fixed threshold is useful in simulation/replay where the
  // intensity distribution is stable; OTSU remains the fallback for live video.
  cv::Mat binary;
  if (config_.binary_threshold > 0.0) {
    cv::threshold(gray, binary, config_.binary_threshold, 255, cv::THRESH_BINARY);
  } else {
    cv::threshold(gray, binary, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);
  }
  if (binary.cols >= 5 && binary.rows >= 5) {
    const cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
    cv::morphologyEx(binary, binary, cv::MORPH_OPEN, kernel);
  }

  // Collect bright points
  std::vector<cv::Point2f> bright_points;
  cv::findNonZero(binary, bright_points);
  for (auto& p : bright_points) {
    p.x += static_cast<float>(roi_int.x);
    p.y += static_cast<float>(roi_int.y);
  }

  if (static_cast<int>(bright_points.size()) < config_.min_bright_points) {
    result.reason = RefineFailReason::TOO_FEW_BRIGHT_POINTS;
    return result;
  }

  // PCA
  cv::Mat data(static_cast<int>(bright_points.size()), 2, CV_32F);
  for (size_t i = 0; i < bright_points.size(); ++i) {
    data.at<float>(static_cast<int>(i), 0) = bright_points[i].x;
    data.at<float>(static_cast<int>(i), 1) = bright_points[i].y;
  }

  cv::PCA pca(data, cv::Mat(), cv::PCA::DATA_AS_ROW);

  float lambda1 = pca.eigenvalues.at<float>(0);
  float lambda2 = pca.eigenvalues.at<float>(1);
  float stability = lambda1 / (lambda1 + lambda2 + 1e-10f);
  if (stability < config_.pca_stability_threshold) {
    result.reason = RefineFailReason::PCA_UNSTABLE;
    return result;
  }

  // Project bright points onto principal axis.
  cv::Point2f mean(pca.mean.at<float>(0), pca.mean.at<float>(1));
  cv::Point2f dir(pca.eigenvectors.at<float>(0, 0),
                  pca.eigenvectors.at<float>(0, 1));
  const float dir_norm = cv::norm(dir);
  if (dir_norm < 1e-6f) {
    result.reason = RefineFailReason::PCA_UNSTABLE;
    return result;
  }
  dir *= 1.0f / dir_norm;
  if (dir.y < 0.0f || (std::abs(dir.y) < 1e-3f && dir.x < 0.0f)) {
    dir *= -1.0f;
  }

  float min_proj = std::numeric_limits<float>::max();
  float max_proj = std::numeric_limits<float>::lowest();
  for (const auto& p : bright_points) {
    float proj = (p.x - mean.x) * dir.x + (p.y - mean.y) * dir.y;
    min_proj = std::min(min_proj, proj);
    max_proj = std::max(max_proj, proj);
  }

  // Extend slightly beyond the PCA projection range.
  float extend = (max_proj - min_proj) * 0.1f;
  cv::Point2f top_pt    = mean + (min_proj - extend) * dir;
  cv::Point2f bottom_pt = mean + (max_proj + extend) * dir;

  // Search for the peak endpoint gradient along the light-bar axis.
  auto searchEdge = [&](cv::Point2f start, cv::Point2f direction,
                         float search_len, int steps) -> cv::Point2f {
    const float direction_norm = cv::norm(direction);
    if (direction_norm < 1e-6f) {
      return start;
    }
    direction *= 1.0f / direction_norm;
    cv::Point2f perp(-direction.y, direction.x);

    cv::Point2f best = start;
    float best_grad = 0.0f;

    for (int s = -steps; s <= steps; ++s) {
      cv::Point2f pt = start + (search_len * s / steps) * direction;

      float grad = 0.0f;
      for (int t = -2; t <= 2; ++t) {
        cv::Point2f p = pt + static_cast<float>(t) * perp;
        const float lx = p.x - static_cast<float>(roi_int.x);
        const float ly = p.y - static_cast<float>(roi_int.y);
        if (lx < 1.0f || lx >= static_cast<float>(gray.cols - 2) ||
            ly < 1.0f || ly >= static_cast<float>(gray.rows - 2)) continue;

        const float before = sampleGrayBilinear(
          gray, lx - direction.x, ly - direction.y);
        const float after = sampleGrayBilinear(
          gray, lx + direction.x, ly + direction.y);
        grad += std::abs(after - before);
      }
      if (grad > best_grad) {
        best_grad = grad;
        best = pt;
      }
    }
    return best;
  };

  float search_range = (max_proj - min_proj) * 0.2f;
  result.top    = searchEdge(top_pt, dir, search_range, 10);
  result.bottom = searchEdge(bottom_pt, dir, search_range, 10);
  if (result.top.y > result.bottom.y) {
    std::swap(result.top, result.bottom);
  }
  result.ok = true;

  return result;
}

bool RoiPcaCornerRefiner::preservesPerspective(
    const std::array<cv::Point2f, 4>& refined,
    const std::array<cv::Point2f, 4>& original) const
{
  if (!config_.preserve_perspective) {
    return true;
  }

  const QuadShape base = computeQuadShape(original);
  const QuadShape candidate = computeQuadShape(refined);
  if (!base.valid || !candidate.valid || base.area < 1e-3) {
    return false;
  }

  if (config_.max_area_ratio_delta > 0.0) {
    const double area_ratio = candidate.area / base.area;
    const double lo = std::max(0.0, 1.0 - config_.max_area_ratio_delta);
    const double hi = 1.0 + config_.max_area_ratio_delta;
    if (area_ratio < lo || area_ratio > hi) {
      return false;
    }
  }

  if (config_.max_length_ratio_delta > 0.0) {
    if (std::abs(candidate.left_right_log_ratio - base.left_right_log_ratio) >
          config_.max_length_ratio_delta ||
        std::abs(candidate.top_bottom_log_ratio - base.top_bottom_log_ratio) >
          config_.max_length_ratio_delta) {
      return false;
    }
  }

  if (config_.max_edge_angle_delta_deg > 0.0) {
    const double max_angle_delta =
      config_.max_edge_angle_delta_deg * CV_PI / 180.0;
    for (size_t i = 0; i < candidate.edge_angles.size(); ++i) {
      if (angleDistance(candidate.edge_angles[i], base.edge_angles[i]) >
          max_angle_delta) {
        return false;
      }
    }
  }

  return true;
}

bool RoiPcaCornerRefiner::validateGeometry(
    const std::array<cv::Point2f, 4>& refined,
    const std::array<cv::Point2f, 4>& original) const
{
  const QuadShape shape = computeQuadShape(refined);
  if (!shape.valid) {
    return false;
  }

  if (shape.area < 20.0 ||
      shape.area > static_cast<double>(frame_cols_ * frame_rows_)) {
    return false;
  }

  if (!cv::isContourConvex(
        std::vector<cv::Point2f>(refined.begin(), refined.end()))) {
    return false;
  }

  if (shape.aspect < config_.min_aspect_ratio ||
      shape.aspect > config_.max_aspect_ratio) {
    return false;
  }

  if (config_.max_side_ratio > 0.0 &&
      shape.side_ratio > std::max(1.0, config_.max_side_ratio)) {
    return false;
  }

  if (config_.max_rectangular_error_deg > 0.0) {
    const cv::Point2f left_center = (refined[0] + refined[1]) * 0.5f;
    const cv::Point2f right_center = (refined[2] + refined[3]) * 0.5f;
    const cv::Point2f left_to_right = right_center - left_center;
    if (cv::norm(left_to_right) < 1e-3f) {
      return false;
    }
    const double roll = std::atan2(left_to_right.y, left_to_right.x);
    const double left_angle = edgeAngle(refined[0], refined[1]);
    const double right_angle = edgeAngle(refined[3], refined[2]);
    const double rect_error = std::max(
      angleDistance(left_angle, roll + CV_PI / 2.0),
      angleDistance(right_angle, roll + CV_PI / 2.0));
    const double max_rect_error =
      config_.max_rectangular_error_deg * CV_PI / 180.0;
    if (rect_error > max_rect_error) {
      return false;
    }
  }

  const bool nn_shape_is_reference =
    config_.method != "sp25_lightbar" &&
    config_.method != "full_lightbar_roi";
  if (nn_shape_is_reference && !preservesPerspective(refined, original)) {
    return false;
  }

  return true;
}

double RoiPcaCornerRefiner::computeQuality(
    const std::array<cv::Point2f, 4>& refined,
    const std::array<cv::Point2f, 4>& original)
{
  // Higher quality when refined keypoints are close to original
  // (avoid large jumps), scaled by consistency of corner positions.
  double total_dist = 0.0;
  for (int i = 0; i < 4; ++i) {
    total_dist += cv::norm(refined[i] - original[i]);
  }
  double avg_dist = total_dist / 4.0;
  return std::max(0.0, 1.0 - avg_dist / 20.0);
}

RefineResult RoiPcaCornerRefiner::refineBySplitPca(
    const cv::Mat& frame,
    const ArmorDetection& detection,
    const std::chrono::steady_clock::time_point& t_start)
{
  RefineResult result;
  result.refined_keypoints = detection.keypoints;

  // Step 1: extract left/right ROI.
  auto [left_roi, right_roi] = extractLightBarROIs(detection.keypoints);

  // Step 2: ROI size gate.
  float min_area = 20.0f;
  if (left_roi.area() < min_area || right_roi.area() < min_area) {
    result.reason = RefineFailReason::ROI_TOO_SMALL;
    return result;
  }

  // Step 3: timeout check.
  if (elapsedMs(t_start) > config_.time_budget_ms) {
    result.reason = RefineFailReason::TIMEOUT;
    return result;
  }

  // Step 4: PCA endpoint refinement for both light bars.
  auto left_ep  = refineLightBar(frame, left_roi, detection.color);
  auto right_ep = refineLightBar(frame, right_roi, detection.color);

  if (!left_ep.ok || !right_ep.ok) {
    result.reason = left_ep.ok ? right_ep.reason : left_ep.reason;
    return result;
  }

  // Step 5: assemble refined keypoints in canonical order:
  // [0]=left_bottom, [1]=left_top, [2]=right_top, [3]=right_bottom
  std::array<cv::Point2f, 4> refined = {{
    left_ep.bottom,
    left_ep.top,
    right_ep.top,
    right_ep.bottom
  }};

  // Step 6: geometry validation.
  if (!validateGeometry(refined, detection.keypoints)) {
    result.reason = RefineFailReason::GEOMETRY_INVALID;
    return result;
  }

  double max_shift = 0.0;
  double total_shift = 0.0;
  for (int i = 0; i < 4; ++i) {
    const double shift = cv::norm(refined[i] - detection.keypoints[i]);
    total_shift += shift;
    max_shift = std::max(max_shift, shift);
  }
  if (config_.max_corner_shift_px > 0.0 &&
      max_shift > config_.max_corner_shift_px) {
    result.reason = RefineFailReason::REFINE_WORSE;
    return result;
  }

  const double mean_shift = total_shift / 4.0;
  if (config_.max_mean_corner_shift_px > 0.0 &&
      mean_shift > config_.max_mean_corner_shift_px) {
    result.reason = RefineFailReason::REFINE_WORSE;
    return result;
  }

  const cv::Point2f refined_center =
    (refined[0] + refined[1] + refined[2] + refined[3]) * 0.25f;
  const double center_shift = cv::norm(refined_center - detection.center);
  if (config_.max_refined_center_shift_px > 0.0 &&
      center_shift > config_.max_refined_center_shift_px) {
    result.reason = RefineFailReason::REFINE_WORSE;
    return result;
  }

  result.refine_quality = computeQuality(refined, detection.keypoints);
  if (config_.min_refine_quality > 0.0 &&
      result.refine_quality < config_.min_refine_quality) {
    result.reason = RefineFailReason::REFINE_WORSE;
    return result;
  }

  // Step 7: success.
  result.ok = true;
  result.refined_keypoints = refined;
  result.elapsed_ms = elapsedMs(t_start);
  result.reason = RefineFailReason::NONE;
  return result;
}

RefineResult RoiPcaCornerRefiner::refine(
    const cv::Mat& frame,
    const ArmorDetection& detection)
{
  const auto t_start = std::chrono::steady_clock::now();
  frame_cols_ = frame.cols;
  frame_rows_ = frame.rows;

  if (config_.method == "sp25_lightbar" ||
      config_.method == "full_lightbar_roi") {
    return refineByLightbarRoi(frame, detection, t_start);
  }

  return refineBySplitPca(frame, detection, t_start);
}

}  // namespace fyt::auto_aim
