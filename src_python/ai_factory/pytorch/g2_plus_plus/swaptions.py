"""Batched exact-transition G2++ swaption Monte Carlo."""
from __future__ import annotations
from time import perf_counter
from typing import Any
import torch
from ai_factory.pytorch.common.device import resolve_device,synchronize,warmup
from ai_factory.pytorch.common.fixed_income import g2_bond_coefficients,g2_path_discount,g2_transition
from ai_factory.pytorch.common.monte_carlo import price_summary
DEFAULT_BATCH_ROWS=128
MAX_PAYMENTS=20
def _column(values,target):return torch.tensor(values,device=target,dtype=torch.float64)[:,None]
def _price_batch_impl(rows,models,curves,products,*,num_paths,target,batch_rows):
    started=perf_counter();outputs=[];payments=torch.arange(1,MAX_PAYMENTS+1,device=target,dtype=torch.float64)[None,:]
    for offset in range(0,len(rows),batch_rows):
        batch=rows[offset:offset+batch_rows];ms=[models[r["model_id"]] for r in batch];cs=[curves[r["curve_id"]] for r in batch];ps=[products[r["product_id"]] for r in batch]
        a=_column([m["mean_reversion_x"] for m in ms],target);sigma=_column([m["volatility_x"] for m in ms],target);b=_column([m["mean_reversion_y"] for m in ms],target);eta=_column([m["volatility_y"] for m in ms],target);rho=_column([m["rho"] for m in ms],target)
        beta0=_column([c["beta0"] for c in cs],target);beta1=_column([c["beta1"] for c in cs],target);beta2=_column([c["beta2"] for c in cs],target);tau=_column([c["tau"] for c in cs],target)
        expiry=_column([p["expiry"] for p in ps],target);accrual=_column([p["accrual_period"] for p in ps],target);strike=_column([p["fixed_rate"] for p in ps],target);notional=_column([p["notional"] for p in ps],target);direction=_column([p["direction"] for p in ps],target);counts=torch.tensor([p["payment_count"] for p in ps],device=target)[:,None]
        maturities=expiry+accrual*payments;ba,bx,by=g2_bond_coefficients(expiry,maturities,a,sigma,b,eta,rho,beta0,beta1,beta2,tau)
        _,_,_,_,chol=g2_transition(a,sigma,b,eta,rho,expiry);normals=torch.randn((len(batch),num_paths,4),device=target,dtype=torch.float64);innovations=torch.matmul(normals,chol.transpose(-1,-2));x,y,ix,iy=innovations.unbind(-1)
        bonds=ba[:,:,None]*torch.exp(-bx[:,:,None]*x[:,None,:]-by[:,:,None]*y[:,None,:]);active=(payments<=counts)[:,:,None];annuity=(active*accrual[:,:,None]*bonds).sum(1);end=torch.gather(bonds,1,(counts-1)[:,:,None].expand(-1,1,num_paths)).squeeze(1)
        discount=g2_path_discount(ix,iy,expiry,a,sigma,b,eta,rho,beta0,beta1,beta2,tau);payoff=discount*notional*torch.clamp(direction*(1-end-strike*annuity),min=0);outputs.extend(price_summary(payoff))
    synchronize(target);return outputs,{"wall_seconds":perf_counter()-started}
def price_batch(rows,model_by_id,curve_by_id,product_by_id,*,num_paths,device,batch_rows=DEFAULT_BATCH_ROWS,**_:Any):
    target=resolve_device(device)
    if target.type=="cuda" and rows:warmup(target,shape=(min(batch_rows,len(rows)),num_paths,4),dtype=torch.float64);_price_batch_impl(rows[:batch_rows],model_by_id,curve_by_id,product_by_id,num_paths=num_paths,target=target,batch_rows=batch_rows)
    return _price_batch_impl(rows,model_by_id,curve_by_id,product_by_id,num_paths=num_paths,target=target,batch_rows=batch_rows)
