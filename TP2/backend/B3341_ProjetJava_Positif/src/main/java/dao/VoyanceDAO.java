/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dao;

import javax.persistence.EntityManager;
import javax.persistence.NoResultException;
import javax.persistence.Query;
import metier.data.Employe;
import metier.data.Voyance;

/**
 *
 * @author ghermet
 */
public class VoyanceDAO {
    
    
    
    public static Voyance chercherVoyance (Employe unEmploye){
        Voyance resultat;
        try{ 
         EntityManager em = JpaUtil.obtenirEntityManager();
         String jpql = "select v from Voyance v where v.employe = :unEmploye AND v.dateFin = null";
         Query query = em.createQuery(jpql);
         query.setParameter("unEmploye", unEmploye);
         resultat = (Voyance) query.getSingleResult();
        }catch(NoResultException e){
            resultat = null;
        }
        return resultat;
    }
    
    public static void modifierVoyance(Voyance v){
        EntityManager em = JpaUtil.obtenirEntityManager();
        if(v!=null){
            em.merge(v);
        }
    }
    
    public static void creerVoyance(Voyance v){
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.persist (v); 
    
    }
    
}
