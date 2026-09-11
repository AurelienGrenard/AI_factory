// Serializes one typed model-parameter row to ordered JSON.
inline nlohmann::ordered_json parameter_json(
    const ModelParameters& parameters
) {
    return {
        $entries
    };
}
