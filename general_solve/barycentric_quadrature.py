def get_quad_pts(qpn):
	wts, alphas, betas = [],[],[]
	if qpn == 1:
		w = 1.000000000000000 
		alpha = 0.333333333333333333333 
		beta = 0.33333333333333333333333 
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)
	if qpn == 2:
		# p= 2 
		w = 0.333333333333333333333333 
		alpha = 0.6666666666666666666666667 
		beta = 0.1666666666666666666666667
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

	if qpn == 3:
		# p= 3
		w = -0.562500000000000 
		alpha =  0.333333333333333333333 
		beta =  0.33333333333333333333333
		wts.append(w)
		alphas.append(alpha)
		betas.append(beta)

		w = 0.520833333333333333333333333  
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
		alpha = 0.333333333333333333333333333 
		beta = 0.33333333333333333333333333 
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

	wts,alphas,betas = grow(wts,alphas,betas)
	return wts, [alphas, betas]

def grow(wts,alphas,betas):
	new_w,new_a,new_b = [],[],[]

	for j in range(len(wts)):
		a = alphas[j]
		b = betas[j]
		print(a,b)
		c = 1-a-b

		ab = abs(a-b) < 1e-10
		ac = abs(a-c) < 1e-10
		bc = abs(b-c) < 1e-10
		check = ab+ac+bc

		if check == 1:
			new_w += [wts[j],wts[j]]
			new_a += [b,c]
			new_b += [c,a]
		elif check == 0:
			new_w += [wts[j]]*5
			new_a += [a,b,b,c,c]
			new_b += [c,a,c,a,b]
		print(new_a,new_b)
	return wts+new_w, alphas+new_a, betas+new_b
