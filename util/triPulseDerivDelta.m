function y = dTriPulseDelta(D, mid, delta);
% vectorized version
% mid: scalar bin center
%   D: can be a matrix
ind1 = (D > mid-delta) & (D <= mid);
ind2 = (D > mid) & (D <= mid+delta);
der = (D - mid)/(delta^2);
y = (ind2 - ind1) * der;
end
