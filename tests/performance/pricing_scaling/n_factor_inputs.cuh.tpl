// Prepare the published seven-factor lift once, independently of paths and launch geometry.
using Prepared = model_binding::PreparedDynamics<7U>;
const float horizon = maximum_maturity_days(products) * kDayFraction;
const auto prepared = model_binding::prepare_dynamics<7U>(models, horizon, kDt);
DeviceBuffer device_prepared(prepared.size() * sizeof(Prepared), DeviceMemoryRole::persistent_input);
copy_to_device(device_prepared, prepared);
