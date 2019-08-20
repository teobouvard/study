/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dao;

import java.util.ArrayList;
import java.util.List;
import javax.persistence.EntityManager;
import javax.persistence.NoResultException;
import javax.persistence.Query;
import static javax.swing.UIManager.get;
import metier.data.Employe;
import metier.data.Medium;
import metier.data.Personne;
import metier.data.Voyance;

/**
 *
 * @author ghermet
 */
public class EmployeDAO {

    public static void ajouterEmploye(Employe e) {
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.persist(e);
    }

    public static void modifierEmploye(Employe e) {
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.merge(e);
    }

    public static List<Employe> getListEmploye() {
        List<Employe> resultat;
        try {
            EntityManager em = JpaUtil.obtenirEntityManager();
            String jpql = "select e From Employe e";
            Query query = em.createQuery(jpql);
            resultat = (List<Employe>) query.getResultList();
        } catch (Exception e) {
            resultat = null;
        }
        return resultat;
    }

    // modif pas testé encore
    public static Employe chercherEmploye(Personne emp) {
        EntityManager em = JpaUtil.obtenirEntityManager();
        Employe e = em.find(Employe.class, emp.getId());
        return e;
    }

    public static Employe selectionnerEmploye(Medium unMedium) {
        List<Employe> resultatList = new ArrayList<Employe>();
        Employe resultat = null;
        try {
            EntityManager em = JpaUtil.obtenirEntityManager();
            String jpql = "select e From Employe e where e.disponible = true";
            Query query = em.createQuery(jpql);
            resultatList = (List<Employe>) query.getResultList();
            boolean noResult = true;
            for (Employe element : resultatList ) {
                if (noResult ==true){
                    if (element.trouverMedium(unMedium)){
                        resultat = element;
                        noResult = false;
                    }
                }
                else if (element.getListVoyance().size() < resultat.getListVoyance().size() && element.trouverMedium(unMedium)) {
                    resultat = element;
                }
            }
        } catch (NoResultException e) {
            resultat = null;
        }
        return resultat;
    }
    
       
}
