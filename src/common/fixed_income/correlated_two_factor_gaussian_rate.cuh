// Exact joint G2 factor/integral step conditional on both Brownian increments.
#pragma once
#include "common/fixed_income/correlated_gaussian_rate.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace ai_factory::workbench::fixed_income {
struct CorrelatedTwoFactorGaussianRateStep {
    CorrelatedGaussianRateStep x, y;
    float residual_lower[10];
};
struct CorrelatedTwoFactorGaussianRateState {
    float x, y, integral;
};

__device__ __forceinline__ CorrelatedTwoFactorGaussianRateStep
prepare_correlated_two_factor_gaussian_rate(
    float ax,float sx,float x0,float ay,float sy,float y0,float rho,float dt) {
    CorrelatedTwoFactorGaussianRateStep out{
        prepare_correlated_gaussian_rate(ax,0.0f,sx,x0,dt),
        prepare_correlated_gaussian_rate(ay,0.0f,sy,y0,dt),{}};
    double c[4][4]{};
    const auto& x=out.x;
    const auto& y=out.y;
    c[0][0]=double(x.state_bridge_loading)*x.state_bridge_loading;
    c[1][0]=c[0][1]=double(x.state_bridge_loading)*x.integral_bridge_loading;
    c[1][1]=double(x.integral_bridge_loading)*x.integral_bridge_loading
        +double(x.integral_second_bridge_loading)*x.integral_second_bridge_loading;
    c[2][2]=double(y.state_bridge_loading)*y.state_bridge_loading;
    c[3][2]=c[2][3]=double(y.state_bridge_loading)*y.integral_bridge_loading;
    c[3][3]=double(y.integral_bridge_loading)*y.integral_bridge_loading
        +double(y.integral_second_bridge_loading)*y.integral_second_bridge_loading;
    // Cross covariances of the four Brownian bridges. An eight-point rule is
    // stable for small a*dt, where closed forms subtract nearly equal values.
    constexpr double nodes[8]={
        0.0198550717512319,0.101666761293187,0.237233795041835,
        0.408282678752175,0.591717321247825,0.762766204958165,
        0.898333238706813,0.980144928248768};
    constexpr double weights[8]={
        0.0506142681451881,0.111190517226687,0.156853322938944,
        0.181341891689181,0.181341891689181,0.156853322938944,
        0.111190517226687,0.0506142681451881};
    const double h=dt;
    const double zx=ax*h,zy=ay*h;
    const double mx=sx>0.0f ? double(x.state_brownian_loading)/(sx*sqrt(h)):0.0;
    const double my=sy>0.0f ? double(y.state_brownian_loading)/(sy*sqrt(h)):0.0;
    const double ix=sx>0.0f ? double(x.integral_brownian_loading)/(sx*sqrt(h)):0.0;
    const double iy=sy>0.0f ? double(y.integral_brownian_loading)/(sy*sqrt(h)):0.0;
    double cross[2][2]{};
    for(unsigned k=0;k<8;++k) {
        const double u=nodes[k];
        const double fx=exp(-zx*u)-mx;
        const double fy=exp(-zy*u)-my;
        const double gx=h*(zx==0.0 ? u : -expm1(-zx*u)/zx)-ix;
        const double gy=h*(zy==0.0 ? u : -expm1(-zy*u)/zy)-iy;
        cross[0][0]+=weights[k]*fx*fy;
        cross[0][1]+=weights[k]*fx*gy;
        cross[1][0]+=weights[k]*gx*fy;
        cross[1][1]+=weights[k]*gx*gy;
    }
    for(unsigned i=0;i<2;++i)
        for(unsigned j=0;j<2;++j) {
            const double v=double(rho)*sx*sy*h*cross[i][j];
            c[i][j+2]=c[j+2][i]=v;
        }
    double l[4][4]{};
    for(unsigned i=0;i<4;++i) {
        for(unsigned j=0;j<=i;++j) {
            double v=c[i][j];
            for(unsigned k=0;k<j;++k) v-=l[i][k]*l[j][k];
            l[i][j]=(i==j) ? sqrt(fmax(0.0,v))
                : (l[j][j]>0.0 ? v/l[j][j] : 0.0);
            out.residual_lower[i*(i+1)/2+j]=static_cast<float>(l[i][j]);
        }
    }
    return out;
}

__device__ __forceinline__ float advance_correlated_two_factor_gaussian_rate(
    const CorrelatedTwoFactorGaussianRateStep& p,const float* z,
    philox::PhiloxKey key,std::uint64_t path,std::uint32_t step,
    std::uint32_t source,CorrelatedTwoFactorGaussianRateState& state) {
    const auto u=philox::domain_uniform_quad(
        key,path,0U,(source<<24U)|step);
    const auto q0=philox::box_muller(u.first,u.second);
    const auto q1=philox::box_muller(u.third,u.fourth);
    const float q[4]={q0.first,q0.second,q1.first,q1.second};
    float r[4]{};
    #pragma unroll
    for(unsigned i=0;i<4;++i)
        for(unsigned j=0;j<=i;++j)
            r[i]=fmaf(p.residual_lower[i*(i+1)/2+j],q[j],r[i]);
    const float increment=p.x.integral_state_loading*state.x
        +p.x.integral_brownian_loading*z[0]+r[1]
        +p.y.integral_state_loading*state.y
        +p.y.integral_brownian_loading*z[1]+r[3];
    state.x=fmaf(p.x.decay,state.x,
        p.x.state_brownian_loading*z[0]+r[0]);
    state.y=fmaf(p.y.decay,state.y,
        p.y.state_brownian_loading*z[1]+r[2]);
    state.integral+=increment;
    return increment;
}
}  // namespace ai_factory::workbench::fixed_income
