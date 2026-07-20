"""Batched exact-transition G2++ Bermudan swaption Monte Carlo."""
from __future__ import annotations
from time import perf_counter
import torch
from ai_factory.pytorch.common.bermudan_lsm import present_value_cashflows
from ai_factory.pytorch.common.device import resolve_device,synchronize,warmup
from ai_factory.pytorch.common.fixed_income import g2_bond_coefficients,g2_path_discount,g2_transition
from ai_factory.pytorch.common.monte_carlo import price_summary
DEFAULT_CPU_BATCH_ROWS=32
DEFAULT_CUDA_BATCH_ROWS=128
MAX_EXERCISES=8
MAX_PAYMENTS=20
BASIS_RATE_SCALE=.04
def _column(values,target):return torch.tensor(values,device=target,dtype=torch.float64)[:,None]
def _price_batch_impl(rows,models,curves,products,*,num_paths,target,batch_rows):
    started=perf_counter();outputs=[];payment_axis=torch.arange(MAX_PAYMENTS,device=target)[None,:];exercise_axis=torch.arange(MAX_EXERCISES,device=target)[None,:]
    for offset in range(0,len(rows),batch_rows):
        batch=rows[offset:offset+batch_rows];ms=[models[r["model_id"]] for r in batch];cs=[curves[r["curve_id"]] for r in batch];ps=[products[r["product_id"]] for r in batch];size=len(batch)
        a=_column([m["mean_reversion_x"] for m in ms],target);sigma=_column([m["volatility_x"] for m in ms],target);b=_column([m["mean_reversion_y"] for m in ms],target);eta=_column([m["volatility_y"] for m in ms],target);rho=_column([m["rho"] for m in ms],target)
        beta0=_column([c["beta0"] for c in cs],target);beta1=_column([c["beta1"] for c in cs],target);beta2=_column([c["beta2"] for c in cs],target);tau=_column([c["tau"] for c in cs],target)
        first=_column([p["first_exercise"] for p in ps],target);period=_column([p["exercise_period"] for p in ps],target);accrual=_column([p["accrual_period"] for p in ps],target);strike=_column([p["fixed_rate"] for p in ps],target);notional=_column([p["notional"] for p in ps],target);direction=_column([p["direction"] for p in ps],target);exercise_count=torch.tensor([p["exercise_count"] for p in ps],device=target);payment_count=torch.tensor([p["payment_count"] for p in ps],device=target)
        times=first+period*exercise_axis;x=torch.zeros((size,num_paths),device=target,dtype=torch.float64);y=torch.zeros_like(x);ix=torch.zeros_like(x);iy=torch.zeros_like(x);states_x=torch.zeros((size,num_paths,MAX_EXERCISES),device=target,dtype=torch.float64);states_y=torch.zeros_like(states_x);discounts=torch.ones_like(states_x);previous=torch.zeros_like(first)
        for e in range(MAX_EXERCISES):
            interval=times[:,e:e+1]-previous;dx,dy,bix,biy,chol=g2_transition(a,sigma,b,eta,rho,interval);innov=torch.matmul(torch.randn((size,num_paths,4),device=target,dtype=torch.float64),chol.transpose(-1,-2));previous_x=x;previous_y=y;x=dx*x+innov[:,:,0];y=dy*y+innov[:,:,1];ix=ix+bix*previous_x+innov[:,:,2];iy=iy+biy*previous_y+innov[:,:,3];states_x[:,:,e]=x;states_y[:,:,e]=y;discounts[:,:,e]=g2_path_discount(ix,iy,times[:,e:e+1],a,sigma,b,eta,rho,beta0,beta1,beta2,tau);previous=times[:,e:e+1]
        immediate=torch.zeros_like(states_x);basis=torch.zeros_like(states_x);maturities=first[:,:,None]+accrual[:,:,None]*(payment_axis[:,None,:]+1)
        for e in range(MAX_EXERCISES):
            ba,bx,by=g2_bond_coefficients(times[:,e:e+1],maturities[:,0,:],a,sigma,b,eta,rho,beta0,beta1,beta2,tau);bonds=ba[:,:,None]*torch.exp(-bx[:,:,None]*states_x[:,None,:,e]-by[:,:,None]*states_y[:,None,:,e]);active=((payment_axis>=e)&(payment_axis<payment_count[:,None]))[:,:,None];annuity=(active*accrual[:,:,None]*bonds).sum(1);end=torch.gather(bonds,1,(payment_count-1)[:,None,None].expand(-1,1,num_paths)).squeeze(1);immediate[:,:,e]=notional*torch.clamp(direction*(1-end-strike*annuity),min=0);basis[:,:,e]=(1-end)/annuity/BASIS_RATE_SCALE
        immediate*= (exercise_axis<exercise_count[:,None])[:,None,:];outputs.extend(price_summary(present_value_cashflows(immediate,basis,discounts,exercise_count)))
    synchronize(target);return outputs,{"wall_seconds":perf_counter()-started}
def price_batch(rows,model_by_id,curve_by_id,product_by_id,*,num_paths,device,batch_rows=None,**_):
    target=resolve_device(device);batch_rows=batch_rows or (DEFAULT_CUDA_BATCH_ROWS if target.type=="cuda" else DEFAULT_CPU_BATCH_ROWS)
    if target.type=="cuda" and rows:warmup(target,shape=(min(batch_rows,len(rows)),num_paths,4),dtype=torch.float64);_price_batch_impl(rows[:batch_rows],model_by_id,curve_by_id,product_by_id,num_paths=num_paths,target=target,batch_rows=batch_rows)
    return _price_batch_impl(rows,model_by_id,curve_by_id,product_by_id,num_paths=num_paths,target=target,batch_rows=batch_rows)
