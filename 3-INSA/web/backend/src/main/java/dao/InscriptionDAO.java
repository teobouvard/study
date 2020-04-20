/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dao;

import javax.persistence.EntityManager;
import metier.data.Personne;

/**
 *
 * @author ghermet
 */
public class InscriptionDAO {
    
        public static Personne chercherPersonne (Integer id, String psw){
        EntityManager em = JpaUtil.obtenirEntityManager();
        Personne p = em.find (Personne.class,id);
        if (p!=null){
            if (!p.getPassword().equals(psw)){
                p = null;
            }
        }
        return p;
    }
    
}
