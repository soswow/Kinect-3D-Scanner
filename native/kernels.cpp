// Dense Kinect image kernels. OpenCV retains ownership of image resampling.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "weighted_fusion.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;

namespace {

template <typename T>
void require_array(const py::array &array, int dimensions, const char *name) {
    if (!array.dtype().equal(py::dtype::of<T>())) {
        throw py::value_error(std::string(name) + " has an unsupported dtype");
    }
    if (array.ndim() != dimensions) {
        throw py::value_error(std::string(name) + " has an unsupported shape");
    }
    if (!(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must be C contiguous");
    }
    if (array.size() && reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T) != 0) {
        throw py::value_error(std::string(name) + " must have aligned storage");
    }
}

// np.rint uses ties-to-even. This implementation does not depend on the
// caller's current floating-point rounding mode, and is used only in bounds.
std::size_t nearest_pixel(double coordinate) {
    const double lower = std::floor(coordinate);
    const double fraction = coordinate - lower;
    const bool round_up = fraction > 0.5 ||
        (fraction == 0.5 && (static_cast<std::size_t>(lower) & 1U) != 0);
    return static_cast<std::size_t>(lower + static_cast<double>(round_up));
}

py::array_t<std::uint16_t> prepare_depth(
    const py::array &depth, double near_mm, double far_mm, bool filter_depth,
    py::ssize_t x0, py::ssize_t y0, py::ssize_t x1, py::ssize_t y1) {
    require_array<std::uint16_t>(depth, 2, "depth");
    if (!std::isfinite(near_mm) || !std::isfinite(far_mm) || near_mm > far_mm) {
        throw py::value_error("Require finite ordered depth bounds");
    }
    const py::ssize_t height = depth.shape(0), width = depth.shape(1);
    if (x1 == -1) x1 = width;
    if (y1 == -1) y1 = height;
    // Reversed resolved slice endpoints describe an empty ROI in NumPy.
    if (x0 < 0 || y0 < 0 || x1 < 0 || y1 < 0 ||
        x0 > width || y0 > height || x1 > width || y1 > height) {
        throw py::value_error("ROI bounds must be resolved within the depth image");
    }
    py::array_t<std::uint16_t> output({height, width});
    if (height == 0 || width == 0) return output;
    const auto *source = static_cast<const std::uint16_t *>(depth.data());
    auto *result = output.mutable_data();
    {
        py::gil_scoped_release release;
        const auto count = static_cast<std::size_t>(height * width);
        // Filtering uses the clipped observation, never already filtered
        // neighbours. Keep a compact snapshot rather than signed image arrays.
        std::vector<std::uint16_t> clipped(filter_depth ? count : 0);
        auto *prepared = filter_depth ? clipped.data() : result;
        for (py::ssize_t y = 0; y < height; ++y) {
            for (py::ssize_t x = 0; x < width; ++x) {
                const py::ssize_t i = y * width + x;
                const auto value = source[i];
                prepared[i] = value > 0 && value >= near_mm && value <= far_mm &&
                    x >= x0 && x < x1 && y >= y0 && y < y1 ? value : 0;
            }
        }
        if (filter_depth) {
            for (py::ssize_t y = 0; y < height; ++y) {
                for (py::ssize_t x = 0; x < width; ++x) {
                    const py::ssize_t i = y * width + x;
                    const int value = prepared[i];
                    if (value == 0) {
                        result[i] = 0;
                        continue;
                    }
                    const double tolerance = std::max(15.0, value * 0.01);
                    unsigned neighbours = 0;
                    const auto agrees = [&](py::ssize_t j) {
                        const int other = prepared[j];
                        return other > 0 && std::abs(other - value) <= tolerance;
                    };
                    if (y > 0) neighbours += agrees(i - width);
                    if (y + 1 < height) neighbours += agrees(i + width);
                    if (x > 0) neighbours += agrees(i - 1);
                    if (x + 1 < width) neighbours += agrees(i + 1);
                    result[i] = neighbours >= 2 ? static_cast<std::uint16_t>(value) : 0;
                }
            }
        }
    }
    return output;
}

py::tuple project_native(
    const py::array &points_rgb, const py::array &depth,
    const py::array &lens, py::ssize_t rgb_height, py::ssize_t rgb_width) {
    require_array<double>(points_rgb, 3, "points_rgb");
    require_array<std::uint16_t>(depth, 2, "depth");
    require_array<double>(lens, 1, "lens");
    const py::ssize_t height = points_rgb.shape(0), width = points_rgb.shape(1);
    if (depth.shape(0) != height || depth.shape(1) != width ||
        points_rgb.shape(2) != 3 || lens.shape(0) != 9) {
        throw py::value_error("Native projection array shapes disagree");
    }
    if (rgb_height < 1 || rgb_width < 1 ||
        rgb_height > std::numeric_limits<py::ssize_t>::max() / rgb_width) {
        throw py::value_error("Invalid RGB image dimensions");
    }
    // Preserve the reference's NumPy/BLAS camera transform accumulation order.
    // Last-bit changes there can move a color sample across an OpenCV remap bin.
    const auto *points_data = static_cast<const double *>(points_rgb.data());
    const auto *depth_data = static_cast<const std::uint16_t *>(depth.data());
    const auto *camera = static_cast<const double *>(lens.data());
    for (int i = 0; i < 9; ++i) {
        if (!std::isfinite(camera[i])) {
            throw py::value_error("Projection parameters must be finite");
        }
    }
    py::array_t<float> map_x({height, width}), map_y({height, width});
    py::array_t<bool> visible({height, width});
    if (height == 0 || width == 0) {
        return py::make_tuple(std::move(map_x), std::move(map_y), std::move(visible));
    }
    auto *out_x = map_x.mutable_data();
    auto *out_y = map_y.mutable_data();
    auto *out_visible = visible.mutable_data();
    {
        py::gil_scoped_release release;
        const auto count = static_cast<std::size_t>(height * width);
        const auto rgb_count = static_cast<std::size_t>(rgb_height * rgb_width);
        std::vector<std::size_t> rgb_index(count);
        std::vector<double> nearest(rgb_count, std::numeric_limits<double>::infinity());
        const double fx = camera[0], fy = camera[1], cx = camera[2], cy = camera[3];
        const double k1 = camera[4], k2 = camera[5], p1 = camera[6], p2 = camera[7], k3 = camera[8];
        for (std::size_t i = 0; i < count; ++i) {
            const double point_x = points_data[i * 3];
            const double point_y = points_data[i * 3 + 1];
            const double point_z = points_data[i * 3 + 2];
            const double inverse_z = point_z != 0.0 ? 1.0 / point_z : 1.0;
            const double x = point_x * inverse_z, y = point_y * inverse_z;
            const double xx = x * x, yy = y * y, xy = x * y;
            const double r2 = xx + yy;
            const double radial = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3));
            const double u = (x * radial + 2.0 * p1 * xy + p2 * (r2 + 2.0 * xx)) * fx + cx;
            const double v = (y * radial + p1 * (r2 + 2.0 * yy) + 2.0 * p2 * xy) * fy + cy;
            out_x[i] = static_cast<float>(u);
            out_y[i] = static_cast<float>(v);
            const bool candidate = depth_data[i] > 0 && point_z > 0.0 &&
                u >= 0.0 && u < rgb_width - 1 && v >= 0.0 && v < rgb_height - 1;
            out_visible[i] = candidate;
            if (candidate) {
                const auto index = nearest_pixel(v) * static_cast<std::size_t>(rgb_width) + nearest_pixel(u);
                rgb_index[i] = index;
                nearest[index] = std::min(nearest[index], point_z);
            }
        }
        // Every originally visible observation contributes to the nearest
        // surface before testing occlusion, matching np.minimum.at semantics.
        for (std::size_t i = 0; i < count; ++i) {
            if (out_visible[i]) {
                const double z = points_data[i * 3 + 2];
                out_visible[i] = z <= nearest[rgb_index[i]] + std::max(15.0, z * 0.01);
            }
        }
    }
    return py::make_tuple(std::move(map_x), std::move(map_y), std::move(visible));
}

}  // namespace

PYBIND11_MODULE(_kinect_native, module) {
    module.doc() = "Strict-array native Kinect depth filtering and RGB projection";
    module.attr("API_VERSION") = 2;
    weighted_fusion::bind_weighted_fusion(module);
    module.def("prepare_depth", &prepare_depth,
        py::arg("depth").noconvert(), py::arg("near_mm"), py::arg("far_mm"),
        py::arg("filter_depth"), py::arg("x0") = 0, py::arg("y0") = 0,
        py::arg("x1") = -1, py::arg("y1") = -1);
    module.def("project_native", &project_native,
        py::arg("points_rgb").noconvert(), py::arg("depth").noconvert(),
        py::arg("lens").noconvert(),
        py::arg("rgb_height"), py::arg("rgb_width"));
}
