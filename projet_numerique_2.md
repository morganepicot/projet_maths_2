## Question 1

On dispose du système différentiel de Lotka-Volterra défini par :
$$
\begin{cases}
\dot x_1 = x_1(\alpha -\beta x_2) \\
\dot x_2 = - x_2( \gamma - \delta x_1) 
\end{cases}
$$
où $x_1$ et $x_2$ définissent respectivement le nombre de proies et de prédateurs dans la simulation étudiée.  
Le paramètre $\alpha$ caractérise ainsi le taux de naissance au sein des proies, tandis que $\beta$ quantifie le taux de proies tuées par les prédateurs.
De la même manière, $\gamma$ donne le taux de mortalité des prédateurs et $\delta$ leur taux d réproduction en fonction des proies mangées.  
On constate immédiatement que le couple $(0, 0)$ est un point fixe du système. Mais le point de coordonnées $(\frac{\gamma}{\delta}, \frac{\alpha}{\beta})$ que l'on notera $\bar x$ aussi.  
On cherche à évaluer leur stabilité. Pour cela, on définit la fonction $f : \R^2 \to\R^2$ telle que $\dot x = f(x)$ par :  
$$
f : x = (x_1, x_2) \to \begin{pmatrix}
x_1(\alpha -\beta x_2) \\
- x_2( \gamma - \delta x_1)
\end{pmatrix}
$$
On calcule alors la différentielle de f : 
$$
\mathrm{d}f(x) = \begin{pmatrix}
\alpha - \beta x_2 & - \beta x_1 \\
\delta x_2 & -( \gamma - \delta x_1)
\end{pmatrix}
$$
On obtient :
$$\mathrm{d} f(0, 0) = \begin{pmatrix}
\alpha & 0 \\ 0 & - \gamma \end{pmatrix}
$$
$(0, 0)$ est instable car sa jacobienne (qui est aussi la différentielle de $f$ en $(0, 0)$) possède une valeur propre strictement positive $\alpha$.  
De plus, 
$$
\mathrm{d}f(\bar x) = \begin{pmatrix}
0 & -\frac{\beta \gamma}{\delta} \\
\frac{\delta \alpha}{\beta} & 0 \end{pmatrix}
$$
Le polynôme caractéristique de cette matrice est $\chi (t) = t^2 + \alpha \gamma$ et les valeurs prpores associées sont $\pm i \sqrt{\alpha \gamma}$. Les parties réelles étant toutes les deux nulles, on ne peut riend dire sur la stabilité ou non du point $\bar x$ pour le système.  


## Question 2
**cf jupyter notebook**

## Question 3

$f$ définie plus haut et dont nous avons calculé la différentielle en question 1 est $\mathcal{C}^1$ par rapport à $x$ sur $\R^2$. Le théorème de Cauchy-Lipschitz s'applique et nous permet d'affirmer que pour une condition initiale $(t_0,x_0)$, la solution maximale de $S_f(t_0,x_0)$ associée est unique.  
Soit $x_0 \in \R_{>0} \times \R_{>0}$.
On considère la solution maximale de $S_f(t_0,x_0)$, notée  
 $x(t) = (x_1(t), x_2(t))$.  
Supposons l'existence de $t_1 \ne t_0$ dans $\mathcal{D}_x$ (domaine de définition de x) tel que $x(t_1) = (a, 0)$ où $a$ est un réel quelconque. $x$ appartient à $S_f(t_1, (a, 0))$ et en est aussi la solution maximale. 
De plus, on peut définir une autre solution $y$ de la forme $y = (y_1, 0)$. Cette solution existe bien et $y_1$ a la forme d'une exponentielle. On choisit alors comme condition initiale $y(t_1) = (a, 0)$. Cette solution est définie pour tout $t$ dans $\R$, donc sur le même intervalle que $x$. C'est aussi une solution maximale pour $S_f(t_1, (a, 0))$.  
On conclut que $x = y$.
Seulement dans ce cas la condition initiale que l'on avait fixée dans $\R_{>0} \times \R_{>0}$ ne sera jamais atteinte par $x$ ce qui est contradictoire.  
Notre hypothèse est donc fausse, il n'existe aucun $t_1$ tel que la coordonnées $x_2$ de $x$ s'annule.
On peut effectuer le même raisonnement avec $x_1$.  
Ainsi, lorsqu'on initialise une solution dans $\R_{>0} \times \R_{>0}$, elle reste dans $\R_{>0} \times \R_{>0}$.

## Question 4

On considère $H$ définie sur $\R_{>0} \times \R_{>0}$ par :  
 $H(x_1, x_2) = \delta x_1 - \gamma \ln (x_1) + \beta x_2 - \alpha \ln(x_2)$.  
 Cette fonction est $\mathcal{C}^1$ sur son ensemble de définition par somme de fonctions $\mathcal{C}^1$ et:
 $$
 \mathrm{d}H(x_1, x_2) = \begin{pmatrix}
 \alpha - \frac{\gamma}{x_1} &
 \beta -\frac{\alpha}{x_2} \end{pmatrix} = \nabla H(x_1, x_2)
 $$

Calculons maintenant $\frac{d}{dt}H(x(t))$.
On a pour tout $x=(x_1, x_2)$ dans $\R_{>0} \times \R_{>0}$:
$$
\frac{d}{dt}H(x(t)) = \langle \nabla H(x), f(t)\rangle = (\delta - \frac{\gamma}{x_1})x_1(\alpha - \beta x_2) - (\beta - \frac{\alpha}{x_2})x_2(\gamma - \delta x_1) = (\gamma - \delta x_1)(\beta x_2 - \alpha) - (\beta x_2 - \alpha)(\gamma - \delta x_1) = 0
$$

Ainsi, H ne dépend pas du temps. Sa norme est identique quel que soit $t$ dans $\R$.
De plus, le système étudié étant autonome, $f$ ne dépend pas explicitement du temps  non plus et est définie pour tout $t$ dans $\R$.  
On considère pour une condition initiale donnée une solution maximale $x$ définie sur l'intervalle ouvert $]t_m^-, t_m^+[$ avec $t_m^-$ et $t_m^+$ dans $\R \cup \{- \infty, +\infty \}$. Supposons $t_m^-$ fini.  
D'après le théorème du domaine maximal d'existence, $x$ explose en temps fini car si elle atteignait la frontière de $f$, $t_m^-$ serait infini par définition. On a donc $\| x \| \to +\infty$ lorsque $t \to t_m^-$.  
Seulement $\| H(x(t)) \| = |H(x(t))| \ge H(x(t))$ et on sait que $\ln (x) \underset{x \to +\infty}{=} \circ (x)$.  
$H$ adopte le même comportement asymptotique que $\delta x_1 + \beta x_2$ en $t_m^-$.  
Ainsi, $\| H\| \to +\infty$ quand $t \to t_m^-$.  
Ce résultat est en contradicton avec celui énoncé plus haut qui veut que $H$, et donc sa norme ne dépendent pas du temps.  
On en conclut que l'hypothèse de départ est fausse : $t_m^-$ ne peut pas être fini.  
Le même raisonnement nous permet de démontrer que $t_m^+$ est également infini. Finalement, $I = \R$.  
Cela montre bien que toute solution initialisée dans $\R_{>0} \times \R_{>0}$ est définie sur $\R$.


 ## Question 5

     
     def H(x1, x2, alpha = alpha,
              beta = beta,
              gamma = gamma,
              delta = delta):
    
    return delta*x1 - gamma*np.log(x1) + beta*x2 - alpha*np.log(x2)

On trace les courbes de niveaux de $H$.

   
    display_contour(
    H, 
    x = np.linspace(0, 100, 100), 
    y = np.linspace(0, 100, 100), 
    levels = 50)
    plt.plot(gamma/delta, alpha/beta, 'r+')
    plt.show()

On constate que $\bar x$, représenté par la croix rouge, se trouve au centre des courbes de niveaux de $H$.



 ## Question 6  
 **cf Claire**

 ## Question 7
 **Idem**

 ## Question 8
   
   
    def point_fixe(F, x_j, t_j, dt, epsilon = 0.05):
    
    X = [x_j, F(x_j, x_j, t_j, dt)]
    
    while np.linalg.norm(X[-1] - X[-2])/np.linalg.norm(X[-2]) > epsilon:
        #print(X[-1])
        X.append(F(x_j, X[-1], t_j, dt))
    
    return X[-1]
    
    def solv_euler_implicit(f, x0, dt, t0, tf, itermax = 10000):
    
    def F(x_j, x, t, dt = dt):
        return x_j + dt*f(x, t + dt)
    
    T = [t0]
    X = [x0]
    n = 0
    
    while T[-1] < tf and n < itermax:
        
        #print(X[-1])
        T.append(T[-1] + dt)
        X.append(point_fixe(F, X[-1], T[-1], dt))
        n += 1
        
    if n == itermax:
        return ('problème avec les itérations')
    
    return T, X

    #on teste pour l'exponentielle
    def f(x, t=0):
    return x
    
    T, X = solv_euler_implicit(f, 1, 0.05, 0, 15)
    plt.plot(T, X, 'r')
    plt.plot(T, np.exp(T), 'b')
    plt.show()
    
    #avec les equations de Lotka-Volterra

    T, X = solv_euler_implicit(LV, np.array([100, 20]), 0.05, 0, 20)
    proies = np.array([v[0] for v in X])
    predateurs = np.array([v[1] for v in X])

    plt.plot(T, proies, 'r')
    plt.plot(T, predateurs, 'b')
    #plt.plot(T, H(proies, predateurs), 'g')
    plt.show()


## Question 9

On a maintenant le système :
$$
\begin{cases}
\dot x_1 = x_1(\alpha -\beta x_2) - u_1(x_1, x_2)(H(x_1, x_2) - H_0)\\
\dot x_2 = - x_2( \gamma - \delta x_1) - u_2(x_1, x_2)(H(x_1, x_2) - H_0)
\end{cases}
$$
On pose $f_2 = f - (H - H_0) \times u$ ce qui permet de l'écrire $\dot x = f_2(x)$.    
  

Si on suppose $H_0 = H(x(0))$, comme $\frac{dH(x(t))}{dt} = 0$, $H(x)$ est constante et on aura 
$$
\forall t \in \mathcal{D}_x, H(x(t)) = H_0
$$  
Mais alors $f_2 = f$ et alors les solutions de $\dot{x} = f_2(x)$ sont les mêmes que celles de $\dot{x} = f(x)$.  
Ce résultat est vrai pour toute fonction $u : \R_2 \times \R_2 \rightarrow \R_2 \times \R_2$, et en particulier si $u$ est continûment différentiable.



## Question 10

On cherche $\frac{d}{dt}(H(x(t)) - H_0)$. Or :
$$
\frac{d}{dt}(H(x) - H_0) = \langle \nabla H(x), f_2(t)\rangle
= \langle \nabla H(x(t)), f(x) \rangle - \langle \nabla H(x), (H(x) - H_0)u(x)
= 0 - (H(x) - H_0) \langle \nabla H(x), u(x) \rangle
$$
Pour $k \in \R$, en choisissant $u : x \to k \nabla H(x)$, on obtient :
$$
\frac{d}{dt}(H(x(t)) - H_0) = - k \| \nabla H(x(t)) \|^2 (H(x(t)) - H_0)
$$
ce qui correspond au résultat souhaité.  
  
Si maintenant $x$ reste à une distance strictement positive de $\bar{x}$, c'est-à-dire qu'on peut trouver $c > 0$ tel que :
$$
\forall t \in \mathcal{D}_x, \|x - \bar{x} \| \geq c >0
$$
On a $\| \nabla H \|^2 = (\delta - \frac{\gamma}{x_1})^2 +(\beta - \frac{\alpha}{x_2})^2$  
et  
$\| x - \bar{x} \|^2 = (\frac{x_1}{\delta})^2(\delta - \frac{\gamma}{x_1})^2 + (\frac{x_2}{\beta})^2(\beta - \frac{\alpha}{x_2})^2$.  

On pose $i = min(\beta, \alpha)$ et on choisit la norme $\|x \| = max(|x_1|, |x_2|)$.  
On a  
$\| x - \bar{x} \|^2 \leq \frac{\|x \|^2}{i^2} \| \nabla H(x) \|^2$.  
D'après la question, $\| x \|$ est bornée sur $\R$, on peut considérer $M > 0$ telle que pour tout $t$,  
$\| x(t) \| \leq M$.
On a donc $\frac{i^2c}{M^2} \leq \| \nabla H(x(t)) \|^2$ quel que soit $t$. On pose $c' = \frac{i^2c}{M^2}$ 
  
En reprenant l'inégalité d'au-dessus, et la croissance de la fonction racine carrée, on a :
$$
\frac{d}{dt}(H(x(t)) - H_0) \leq - kc'(H(x(t)) - H_0)
$$
Par positivité de la fonction exponentielle on peut multiplier des deux côtés par $e^{kc't}$ les deux membres. On a alors :
$$
\forall t \in \R,  \frac{d}{dt}(e^{kc't}(H(x(t)) - H_0)) \leq 0
$$
Par croissance de l'intégrale on a $a \in \R$ tel que
$$
\forall t \in \R, H(x(t)) - H_0 \leq ae^{-kc't}
$$
De la même manière, en multipliant l'égalité de départ par $-1$, on peut refaire tout le raisonnement avec $H_0 - H(x(t))$ au lieu de $H(x(t)) - H_0$. On aboutit à :
$$
\forall t \in \R,  \frac{d}{dt}(e^{kc't}(H_0 - H(x(t)))) \leq 0
$$
On a $b \in \R$ tel que
$$
\forall t \in \R, H_0 - H(x(t))\leq be^{-kc't}
$$
Finalement,
$$
\forall t \in \R, -be^{-kc't} \leq H(x(t)) - H_0 \leq ae^{-kc't}
$$
On a bien démontré que $H(x)$ converge exponentiellement vers $H_0$ quand $t$ tend vers l'infini par encadrement.

## Question 11

**C'EST LA DERNIEEEEEEERE**