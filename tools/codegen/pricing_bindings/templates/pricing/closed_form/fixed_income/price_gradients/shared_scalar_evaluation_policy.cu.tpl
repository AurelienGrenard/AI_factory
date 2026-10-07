// Share one scalar Jamshidian evaluation body between sensitivity kernels.
namespace {
template<typename Base>
struct SharedScalarEvaluationPolicy : Base {
    using Base::evaluate;
    using InputRow = typename Base::InputRow;

    // Keep Jamshidian's scalar root solve identical across kernels.
    __device__ __noinline__ static float evaluate(const InputRow& input) {
        return Base::evaluate(input);
    }
};
}  // namespace
