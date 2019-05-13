/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dao;

import java.util.List;
import javax.persistence.EntityManager;
import javax.persistence.Query;
import metier.data.Medium;

/**
 *
 * @author ghermet
 */
public class MediumDAO {
    
    public static Medium chercherMedium (Integer id){
        EntityManager em = JpaUtil.obtenirEntityManager();
        Medium m = em.find (Medium.class,id);
        return m;
    }
    
    public static void ajouterMedium ( Medium m ){
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.persist (m); 
    }
    
    public static void modifierMedium(Medium m){
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.merge(m);
    }
    
    public static List<Medium> getListMedium(){
     List<Medium> resultat;
        try{ 
         EntityManager em = JpaUtil.obtenirEntityManager();
         String jpql = "select m From Medium m";
         Query query = em.createQuery(jpql);
         resultat = (List<Medium>) query.getResultList();
        }catch(Exception e){
            resultat = null;
        }
        return resultat;
    }  
    
}
