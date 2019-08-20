/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.service;

import dao.EmployeDAO;
import dao.JpaUtil;
import dao.MediumDAO;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import javax.persistence.RollbackException;
import metier.data.Astrologue;
import metier.data.Employe;
import metier.data.Medium;
import metier.data.Tarologue;
import metier.data.Voyant;

/**
 *
 * @author ghermet
 */
public class ServiceInit {

    public static void init() throws ParseException {
 
        Tarologue m1 = new Tarologue("Mme Irma", "Voyance de qualité");
        Tarologue m2 = new Tarologue("Mme Soleil", "Apporte la lumière");
        Voyant m3 = new Voyant("Mage Billot","les boules de cristal", "C'est pas une bille");
        Voyant m4 = new Voyant("Mr Shipton","les cartes", "Trouve les as");
        Astrologue m5 = new Astrologue("Mme Bellune","Toujours dans la lune","2021", "INSA");
        
        SimpleDateFormat JSON_DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd");
        Employe e1 = new Employe("Mr", "Dupont", "Jean", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des oliviers", "j.dupont@gmail.com", "0102030405", "jdupont");
        Employe e2 = new Employe("Mr", "Charles", "Magnes", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des caribous", "m.charles@gmail.com", "0102030405", "mcharles");
        Employe e3 = new Employe("Mr", "Francois", "Premier", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des violettes", "p.francois@gmail.com", "0102030405", "pfrancois");
        Employe e4 = new Employe("Mr", "Louis", "Seize", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des albatros", "s.louis@gmail.com", "0102030405", "slouis");
        Employe e5 = new Employe("Mr", "Capet", "Hugues", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des tapis", "h.capet@gmail.com", "0102030405", "hcapet");
        
        //on ajoute les medium et employe avec leur liste vide
        ajouterMedium(m1);
        ajouterMedium(m2);
        ajouterMedium(m3);
        ajouterMedium(m4);
        ajouterMedium(m5);
        
        ajouterEmploye(e1);
        ajouterEmploye(e2);
        ajouterEmploye(e3);
        ajouterEmploye(e4);
        ajouterEmploye(e5);
        
        
        // on modifie les employes pour leur r'ajouter leur liste
        
        e1.ajouterMedium(m1);
        e1.ajouterMedium(m3);
        m1.ajouterEmploye(e1);
        m3.ajouterEmploye(e1);
        
        e2.ajouterMedium(m2);
        e2.ajouterMedium(m4);
        m2.ajouterEmploye(e2);
        m4.ajouterEmploye(e2);
        
        e3.ajouterMedium(m4);
        e3.ajouterMedium(m5);
        m4.ajouterEmploye(e3);
        m5.ajouterEmploye(e3);
        
        e4.ajouterMedium(m1);
        e4.ajouterMedium(m5);
        m1.ajouterEmploye(e4);
        m5.ajouterEmploye(e4);
        
        e5.ajouterMedium(m3);
        e5.ajouterMedium(m4);
        m3.ajouterEmploye(e5);
        m4.ajouterEmploye(e5);
        // on va persister les modifications dans la table
        
        modifierMedium(m1);
        modifierMedium(m2);
        modifierMedium(m3);
        modifierMedium(m4);
        modifierMedium(m5);
        
        modifierEmploye(e1);
        modifierEmploye(e2);
        modifierEmploye(e3);
        modifierEmploye(e4);
        modifierEmploye(e5);

        

    }

    public static boolean ajouterEmploye(Employe emp) {
        boolean opReussie = true;
        try {
            JpaUtil.creerEntityManager();
            JpaUtil.ouvrirTransaction();
            EmployeDAO.ajouterEmploye(emp);
            JpaUtil.validerTransaction();
        } catch (RollbackException e) {
            JpaUtil.annulerTransaction();
            opReussie = false;
        } finally {
            JpaUtil.fermerEntityManager();
        }
        return opReussie;
    }
    
        public static boolean ajouterMedium(Medium med) {
            boolean opReussie = true;
            try {
                JpaUtil.creerEntityManager();
                JpaUtil.ouvrirTransaction();
                MediumDAO.ajouterMedium(med);
                JpaUtil.validerTransaction();
            } catch (RollbackException e) {
                JpaUtil.annulerTransaction();
                opReussie = false;
            } finally {
                JpaUtil.fermerEntityManager();
            }
            return opReussie;
    }
        
        
        
        
    public static boolean modifierEmploye(Employe emp) {
        boolean opReussie = true;
        try {
            JpaUtil.creerEntityManager();
            JpaUtil.ouvrirTransaction();
            EmployeDAO.modifierEmploye(emp);
            JpaUtil.validerTransaction();
        } catch (RollbackException e) {
            JpaUtil.annulerTransaction();
            opReussie = false;
        } finally {
            JpaUtil.fermerEntityManager();
        }
        return opReussie;
    }
    public static boolean modifierMedium(Medium med) {
        boolean opReussie = true;
        try {
            JpaUtil.creerEntityManager();
            JpaUtil.ouvrirTransaction();
            MediumDAO.modifierMedium(med);
            JpaUtil.validerTransaction();
        } catch (RollbackException e) {
            JpaUtil.annulerTransaction();
            opReussie = false;
        } finally {
            JpaUtil.fermerEntityManager();
        }
        return opReussie;
    }
}
