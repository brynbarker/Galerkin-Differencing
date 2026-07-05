def get_quad_pts(qpn):
	wts, alphas, betas = [],[],[]
	if qpn == 1:
		w = 1.000000000000000 
		alpha = 0.333333333333333 
		beta = 0.333333333333333 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)
	if qpn == 2:
		# p= 2 
		w = 0.333333333333333 
		alpha = 0.666666666666667 
		beta = 0.166666666666667
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

	if qpn == 3:
		# p= 3
		w = -0.562500000000000 
		alpha =  0.333333333333333 
		beta =  0,333333333333333
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.520833333333333  
		alpha = 0.600000000000000  
		beta = 0.200000000000000 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

	if qpn == 4:
		# p= 4 
		w = 0.223381589678011 
		alpha = 0.108103018168070 
		beta = 0.445948490915965 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.109951743655322 
		alpha = 0.816847572980459 
		beta = 0.091576213509771 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

	if qpn == 5:
		# p= 5 
		w = 0.225000000000000 
		alpha = 0.333333333333333 
		beta = 0.333333333333333 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.132394152788506 
		alpha = 0.059715871789770 
		beta = 0.470142064105115
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.125939180544827 
		alpha = 0.797426985353087 
		beta = 0.101286507323456 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

	if qpn == 6:
		# p= 6 
		w = 0.116786275726379 
		alpha = 0.501426509658179 
		beta = 0.249286745170910 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.050844906370207 
		alpha = 0.873821971016996 
		beta = 0.063089014491502 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.082851075618374 
		alpha = 0.053145049844817 
		beta = 0.310352451033784 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

	return wts, alphas, betas