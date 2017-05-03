# Various Julia implementations of SOUP-DIL[LO].

module SOUPDIL

export soupdillo

"""
    soupdillo(Y::Matrix{Float64},J::Int,λ::Float64,K::Int,L::Float64)

Compute a synthesis dictionary and associated sparse coefficients for
the columns of Y using the SOUP-DIL[L0] algorithm.
"""
function soupdillo(Y::Matrix{Float64},J::Int,λ::Float64,K::Int,L::Float64)

  # Useful definitions
  n,N = size(Y)
  v₁ = eye(n)[:,1]

  # Initial estimates
  D = gen_odct(n,J)
  C = zeros(N,J)

  b₁ = Vector{Float64}(N)
  b₂ = Vector{Float64}(J)
  b₃ = Vector{Float64}(N)
  h₁ = Vector{Float64}(n)
  h₂ = Vector{Float64}(J)
  h₃ = Vector{Float64}(n)
  for t = 1:K
    for j = 1:J
      # 1) C = [c_1^t,...,c_{j-1}^t,c_j^{t-1},...,c_J^{t-1}]
      #    D = [d_1^t,...,d_{j-1}^t,d_j^{t-1},...,d_J^{t-1}]

      # 2) Sparse coding
      @inbounds At_mul_B!(b₁,Y,D[:,j])
      @inbounds At_mul_B!(b₂,D,D[:,j])
      A_mul_B!(b₃,C,b₂)
      @inbounds bₜ = b₁ - b₃ + C[:,j]
      for i in eachindex(bₜ)
        @inbounds abs(bₜ[i]) < λ && (bₜ[i] = 0)
        @inbounds abs(bₜ[i]) > L && (bₜ[i] = L*sign(bₜ[i]))
      end
      cⱼₜ = bₜ

      # 3) Dictionary atom update
      A_mul_B!(h₁,Y,cⱼₜ)
      At_mul_B!(h₂,C,cⱼₜ)
      A_mul_B!(h₃,D,h₂)
      @inbounds hₜ  = h₁ - h₃ + D[:,j]*dot(C[:,j],cⱼₜ)
      dⱼₜ = !iszero(cⱼₜ) ? hₜ/norm(hₜ,2) : v₁

      @inbounds C[:,j] = cⱼₜ
      @inbounds D[:,j] = dⱼₜ
    end
  end

  return D, C
end

function iszero(B)
  for b in B
    b != 0 && (return false)
  end
  return true
end

function gen_odct(n::Int,J::Int)
  sqrtn,sqrtJ = ceil(Int,sqrt([n,J]))

  odct = zeros(sqrtn,sqrtJ)
  odct[:,1] = 1. ./ sqrt(sqrtn)
  for j = 2:sqrtJ
    v = cos(π*(j-1)/sqrtJ * (0:sqrtn-1))'; v -= mean(v)
    odct[:,j] = v/norm(v,2)
  end
  odct = kron(odct,odct)

  return odct[1:n,1:J]
end

end
