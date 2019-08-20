/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package metier.service;

import dao.ClientDAO;
import dao.EmployeDAO;
import dao.InscriptionDAO;
import dao.JpaUtil;
import dao.MediumDAO;
import dao.VoyanceDAO;
import java.io.IOException;
import java.util.Date;
import java.util.List;
import javax.persistence.RollbackException;
import metier.data.Client;
import metier.data.Employe;
import metier.data.Medium;
import metier.data.Personne;
import metier.data.Voyance;
import util.AstroTest;
import util.Message;

/**
 *
 * @author ghermet
 */
public class Service {
    
    // services inscription et login
    public static boolean ajouterClient(Client c) {
        boolean opReussie = true;
        try {
            JpaUtil.creerEntityManager();
            JpaUtil.ouvrirTransaction();
            ClientDAO.AjouterClient(c);
            JpaUtil.validerTransaction();
        } catch (RollbackException e) {
            JpaUtil.annulerTransaction();
            opReussie = false;
        } finally {
            JpaUtil.fermerEntityManager();
        }
        if (opReussie){
            String corps = "Bonjour " + c.getPrenom() + ", nous vous confirmons votre inscription au service POSIT'IF. Votre numero de client est: " + c.getId() + ".";
            Message.envoyerMail("contact@posit.fr", c.getAdresseMail(), "Bienvenu chez POSIT'IF", corps);
        }else{
            
             String corps = "Bonjour " + c.getPrenom() + ", nous vous informons que votre inscription au service POSIT'IF a malencontreusement échoué... Merci de recommencer ultérieurement.";
            Message.envoyerMail("contact@posit.fr", c.getAdresseMail(), "Bienvenu chez POSIT'IF" , corps);  
        
        }
        return opReussie;
    }
    
    
    public static Personne chercherPersonne(Integer id, String psw){
        JpaUtil.creerEntityManager();
        Personne p = InscriptionDAO.chercherPersonne(id, psw);
        JpaUtil.fermerEntityManager();
        return p;
    }
    
    public static Client chercherClient(Personne p){
        JpaUtil.creerEntityManager();
        Client c = ClientDAO.chercherClient(p);
        JpaUtil.fermerEntityManager();
        return c;
    }
    
    public static Employe chercherEmploye(Personne p){
        JpaUtil.creerEntityManager();
        Employe e = EmployeDAO.chercherEmploye(p);
        JpaUtil.fermerEntityManager();
        return e;
    }


// Service Employe
    public static Voyance chercherVoyance (Employe unEmploye){
        JpaUtil.creerEntityManager(); 
        Voyance v = VoyanceDAO.chercherVoyance (unEmploye);
        JpaUtil.fermerEntityManager();
        return v;
    }
    
    // Service Employe
    public static void demarrerVoyance (Voyance v)
    {
        String msg = "Votre demande de voyance du " + v.getDemandeVoyance() + " a bien été enregistrée. Vous pouvez à present me contacter au " + v.getEmploye().getNumero() + ". A tout de suite! Posit'ifement votre, " + v.getMedium().getNom() ; 
        Message.envoyerNotification(v.getClient().getNumero(), msg);
    }
// Service Employe
    public static List<String> donnerPrediction (Client c, int amour, int sante, int travail) throws IOException{
           AstroTest astroAPI = new AstroTest();
           
           return astroAPI.getPredictions(c.getCouleur(), c.getAnimal(), amour, sante, travail);
    }
    // Service Employe
    public static boolean endVoyance(Voyance v,Date debut,Date fin, String com){
        boolean opReussie = true;
        v.setDateDebut(debut);
        v.setCommentaire(com);
        v.setDateFin(fin);
        v.getEmploye().setDisponible(true);
        v.getClient().setDemandeFaite(false);

        try {
            JpaUtil.creerEntityManager();
            JpaUtil.ouvrirTransaction();
            VoyanceDAO.modifierVoyance(v);
            ClientDAO.modifierClient(v.getClient());
            EmployeDAO.modifierEmploye(v.getEmploye());
            JpaUtil.validerTransaction();
        } catch (RollbackException e) {
            JpaUtil.annulerTransaction();
            opReussie = false;
        } finally {
            JpaUtil.fermerEntityManager();
        }
        return opReussie;
    }
    
     // Service Employe
    public static List<Employe> getListEmploye(){
    JpaUtil.creerEntityManager(); 
        List <Employe> e = EmployeDAO.getListEmploye();
    JpaUtil.fermerEntityManager();
    return e;
    }
    
   // Service Client 
    public static List<Medium> getListMedium(){
    JpaUtil.creerEntityManager(); 
        List <Medium> m = MediumDAO.getListMedium();
    JpaUtil.fermerEntityManager();
    return m;
    }
    
       
// services Clients
    public static Medium chercherMedium (Integer id){
        JpaUtil.creerEntityManager();
        Medium m = MediumDAO.chercherMedium(id);
        JpaUtil.fermerEntityManager();
        return m;
    }


// sous methodes de demanderVoyance
// pour donner l'employé disponible et avec le moins de voyance pour jouer un medium
public static Employe selectionnerEmploye(Medium unMedium){ 
    JpaUtil.creerEntityManager();
    Employe e = EmployeDAO.selectionnerEmploye(unMedium);
    JpaUtil.fermerEntityManager();
    return e;
}

public static void notifierEmploye(Voyance v){
    String msg = "Voyance demandée le " + v.getDemandeVoyance() + " pour " + v.getClient().getPrenom() + " " + v.getClient().getNom() + "(#" + v.getClient().getId() + "). Medium a incarner : " + v.getMedium().getNom();
     Message.envoyerNotification(v.getClient().getNumero(), msg);
}

// cree une voyance, envoi une notif a l'employer qui ferra la voyance, renvoi un null si aucune voyance n'est possible.
public static Voyance demanderVoyance(Medium unMedium, Client unClient){
    
    Date d = new Date();
    Voyance v = new Voyance(d, unClient, unMedium);
    Employe emp = selectionnerEmploye(unMedium);
    if (emp != null && unClient.getDemandeFaite()==false){
        emp.setDisponible(false);
        unClient.setDemandeFaite(true);
       
        v.setEmploye(emp);
        emp.ajouterVoyance(v);
        notifierEmploye(v);
        
        unClient.ajouterVoyance(v);
        unMedium.ajouterVoyance(v);
        
        try{
            JpaUtil.creerEntityManager();
            JpaUtil.ouvrirTransaction();
            VoyanceDAO.creerVoyance(v);
            EmployeDAO.modifierEmploye(emp);
            ClientDAO.modifierClient(unClient);
            MediumDAO.modifierMedium(unMedium);
            JpaUtil.validerTransaction();
        }catch (RollbackException e){
            JpaUtil.annulerTransaction();
        }
        finally{
            JpaUtil.fermerEntityManager();
        }
    }else {
        v = null;
    }
   
    
    return v;
}

}



