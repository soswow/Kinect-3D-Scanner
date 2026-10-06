#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Keep the camera transform in NumPy: its BLAS accumulation order can affect
// which pixel a boundary voxel observes. This kernel fuses the subsequent
// projection, gates, weighted average, and scatter without Python temporaries.
namespace weighted_fusion {
namespace py = pybind11;

template <typename T>
inline void check_array(const py::array& array, const char* name, int ndim,
                        bool writable = false) {
    if (!array.dtype().equal(py::dtype::of<T>())) {
        throw py::type_error(std::string(name) + " has an unexpected dtype");
    }
    if (array.ndim() != ndim || !(array.flags() & py::array::c_style)) {
        throw py::value_error(std::string(name) + " must have the required shape and be C-contiguous");
    }
    if (writable && !array.writeable()) {
        throw py::value_error(std::string(name) + " must be writable");
    }
    if (array.size() && reinterpret_cast<std::uintptr_t>(array.data()) % alignof(T)) {
        throw py::value_error(std::string(name) + " must have aligned storage");
    }
}

inline bool overlaps(const py::array& a, const py::array& b) {
    if (a.size() == 0 || b.size() == 0) {
        return false;
    }
    const auto a_start = reinterpret_cast<std::uintptr_t>(a.data());
    const auto b_start = reinterpret_cast<std::uintptr_t>(b.data());
    const auto a_size = static_cast<std::uintptr_t>(a.nbytes());
    const auto b_size = static_cast<std::uintptr_t>(b.nbytes());
    if (a_start > std::numeric_limits<std::uintptr_t>::max() - a_size ||
        b_start > std::numeric_limits<std::uintptr_t>::max() - b_size) {
        throw py::value_error("Array storage address range overflows");
    }
    return a_start < b_start + b_size && b_start < a_start + a_size;
}

struct Update {
    std::int64_t id;
    float tsdf;
    float weight;
    float color[3];
};

inline std::size_t integrate_cpu(
    const py::array& xyz, const py::array& indices, const py::array& depth,
    const py::array& confidence, const py::array& rgb, py::array tsdf,
    py::array weight, py::array color, float fx, float fy, float cx, float cy,
    float max_depth, float truncation) {
    check_array<float>(xyz, "xyz", 2);
    check_array<std::int64_t>(indices, "indices", 1);
    check_array<float>(depth, "depth", 2);
    check_array<float>(confidence, "confidence", 2);
    check_array<std::uint8_t>(rgb, "rgb", 3);
    check_array<float>(tsdf, "tsdf", 2, true);
    check_array<float>(weight, "weight", 2, true);
    check_array<float>(color, "color", 2, true);
    const auto count = xyz.shape(0);
    const auto rows = depth.shape(0);
    const auto columns = depth.shape(1);
    const auto capacity = tsdf.shape(0);
    if (xyz.shape(1) != 3 || indices.shape(0) != count || rows <= 0 || columns <= 0 ||
        confidence.shape(0) != rows || confidence.shape(1) != columns ||
        rgb.shape(0) != rows || rgb.shape(1) != columns || rgb.shape(2) != 3 ||
        tsdf.shape(1) != 1 || weight.shape(0) != capacity || weight.shape(1) != 1 ||
        color.shape(0) != capacity || color.shape(1) != 3) {
        throw py::value_error("Weighted fusion array shapes do not match");
    }
    if (!std::isfinite(fx) || fx <= 0 || !std::isfinite(fy) || fy <= 0 ||
        !std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(max_depth) || max_depth <= 0 ||
        !std::isfinite(truncation) || truncation <= 0) {
        throw py::value_error("Weighted fusion camera, depth, and truncation parameters must be finite and valid");
    }
    const std::array<py::array, 5> inputs = {xyz, indices, depth, confidence, rgb};
    const std::array<py::array, 3> outputs = {tsdf, weight, color};
    for (std::size_t i = 0; i < outputs.size(); ++i) {
        for (const auto& input : inputs) {
            if (overlaps(outputs[i], input)) {
                throw py::value_error("Weighted fusion inputs and attributes must not overlap");
            }
        }
        for (std::size_t j = i + 1; j < outputs.size(); ++j) {
            if (overlaps(outputs[i], outputs[j])) {
                throw py::value_error("Weighted fusion attributes must not overlap");
            }
        }
    }

    const auto* positions = static_cast<const float*>(xyz.data());
    const auto* ids = static_cast<const std::int64_t*>(indices.data());
    const auto* observed_depth = static_cast<const float*>(depth.data());
    const auto* incoming_weight = static_cast<const float*>(confidence.data());
    const auto* image_color = static_cast<const std::uint8_t*>(rgb.data());
    auto* tsdf_data = static_cast<float*>(tsdf.mutable_data());
    auto* weight_data = static_cast<float*>(weight.mutable_data());
    auto* color_data = static_cast<float*>(color.mutable_data());

    std::size_t integrated = 0;
    {
        py::gil_scoped_release release;
        // Validate every index before changing shared Open3D storage, including
        // indices for samples subsequently excluded by the observation gates.
        for (py::ssize_t i = 0; i < count; ++i) {
            if (ids[i] < 0 || ids[i] >= capacity) {
                throw std::out_of_range("Weighted fusion voxel index is outside attribute storage");
            }
        }
        std::vector<Update> updates;
        updates.reserve(static_cast<std::size_t>(count));
        for (py::ssize_t i = 0; i < count; ++i) {
            const float z = positions[3 * i + 2];
            if (!(z > 0)) {
                continue;
            }
            const float safe_z = z < 1e-6f ? 1e-6f : z;
            const float projected_u = positions[3 * i] * fx / safe_z + cx;
            const float projected_v = positions[3 * i + 1] * fy / safe_z + cy;
            if (!std::isfinite(projected_u) || !std::isfinite(projected_v)) {
                continue;
            }
            // Match the reference's float32 abs + 0.5 then floor. std::round
            // differs just below some half-pixel boundaries after this sum
            // rounds to float32. Avoid casting enormous projections to int64.
            const float rounded_u = std::copysign(std::floor(std::abs(projected_u) + 0.5f), projected_u);
            const float rounded_v = std::copysign(std::floor(std::abs(projected_v) + 0.5f), projected_v);
            if (rounded_u < 0 || rounded_v < 0 ||
                static_cast<double>(rounded_u) >= static_cast<double>(columns) ||
                static_cast<double>(rounded_v) >= static_cast<double>(rows)) {
                continue;
            }
            const auto u = static_cast<py::ssize_t>(rounded_u);
            const auto v = static_cast<py::ssize_t>(rounded_v);
            const auto pixel = v * columns + u;
            const float observed = observed_depth[pixel];
            const float incoming = incoming_weight[pixel];
            const float sdf = observed - z;
            if (!(observed > 0) || !(observed <= max_depth) ||
                !(incoming > 0) || !(sdf >= -truncation)) {
                continue;
            }
            const auto id = ids[i];
            const float old_weight = weight_data[id];
            const float total = old_weight + incoming;
            const float distance = std::min(sdf / truncation, 1.0f);
            Update update;
            update.id = id;
            update.tsdf = (tsdf_data[id] * old_weight + distance * incoming) / total;
            update.weight = total;
            for (int channel = 0; channel < 3; ++channel) {
                update.color[channel] = (color_data[3 * id + channel] * old_weight +
                    static_cast<float>(image_color[3 * pixel + channel]) * incoming) / total;
            }
            updates.push_back(update);
        }
        // Read all old attributes before writing, matching NumPy's advanced
        // assignment even when indices repeat (the last valid sample wins).
        for (const auto& update : updates) {
            tsdf_data[update.id] = update.tsdf;
            weight_data[update.id] = update.weight;
            for (int channel = 0; channel < 3; ++channel) {
                color_data[3 * update.id + channel] = update.color[channel];
            }
        }
        integrated = updates.size();
    }
    return integrated;
}

inline void bind_weighted_fusion(py::module_& module) {
    module.def(
        "integrate_weighted_cpu", &integrate_cpu,
        py::arg("xyz").noconvert(), py::arg("indices").noconvert(),
        py::arg("depth").noconvert(), py::arg("confidence").noconvert(),
        py::arg("rgb").noconvert(), py::arg("tsdf").noconvert(),
        py::arg("weight").noconvert(), py::arg("color").noconvert(),
        py::arg("fx"), py::arg("fy"), py::arg("cx"), py::arg("cy"),
        py::arg("max_depth"), py::arg("truncation"),
        "Update writable shared CPU TSDF attributes from float32 camera-space voxels; return the number of integrated samples.");
}
}  // namespace weighted_fusion
