// Load the canonical curve composition; its rows stay aligned with model and product rows.
using Curve = curve::$CURVE::$CURVE_TYPE;
const auto curves = curve::$CURVE::load_curves(input_paths.curve);
if (curves.size() != models.size()) throw std::invalid_argument("Curve rows are not aligned.");
DeviceBuffer device_curves(curves.size() * sizeof(Curve), DeviceMemoryRole::persistent_input);
copy_to_device(device_curves, curves);
