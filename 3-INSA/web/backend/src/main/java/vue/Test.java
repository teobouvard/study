/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package vue;

import dao.JpaUtil;
import java.io.IOException;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.List;
import metier.data.Client;
import metier.data.Employe;
import metier.data.Medium;
import metier.data.Personne;
import metier.data.Voyance;
import metier.service.Service;
import metier.service.ServiceInit;
import static util.AstroTest.JSON_DATE_FORMAT;
import util.Saisie;

/**
 *
 * @author ghermet
 */
public class Test {
    public static void main (String [] args) throws IOException, ParseException{
        JpaUtil.init();
        ServiceInit.init();
        int choix = 1;
        while (choix !=0){
            System.out.println("Que voulez-vous faire ?");
            System.out.println("1 - Tester l'ajout de clients");
            System.out.println("2 - Tester la recherche de personne");
            System.out.println("3 - Tester la demande de voyance");
            System.out.println("4 - Tester la recherche de voyance");
            System.out.println("5 - Tester le démarrage de la voyance");
            System.out.println("6 - Tester les prédictions sur un client");
            System.out.println("7 - Tester la finalisation de la voyance");
            System.out.println("8 - Tester le retour de la liste d'employés");
            System.out.println("9 - Tester le retour de la liste de médiums");
            System.out.println("0 - Quitter");

            choix = Saisie.lireInteger("Entrez votre choix :");

            switch (choix){
                case 0 : break;
                case 1 : testAjouterClient();break;
                case 2 : testChercherPersonne();break;
                case 3 : testDemanderVoyance();break;
                case 4 : testChercherVoyance();break;
                case 5 : testDemarrerVoyance();break;
                case 6 : testDonnerPrediction();break;
                case 7 : testEndVoyance();break;
                case 8 : testGetListEmploye();break;
                case 9 : testGetListMedium();break;
                default : System.out.println("Entrer un choix valide");
            }
        }
        JpaUtil.destroy();
    }
    
    public static void testAjouterClient() throws IOException, ParseException{
        System.out.println("début de testAjouterClient");
        SimpleDateFormat JSON_DATE_FORMAT = new SimpleDateFormat("yyyy-MM-dd");
        Client c1 = new Client ("Mr", "Escobar", "Pablo", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des lilas", "p.esco@gmail.com", "0657482121", "narco");
        Client c2 = new Client ("Mr", "Guzman", "Joackim", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des alouettes", "j.guzman@gmail.com", "0657482121", "narco");
        Client c3 = new Client ("Mr", "Escobar2", "Pablo2", JSON_DATE_FORMAT.parse("1975-12-02"), "rue des lilas", "p.esco@gmail.com", "0657482121", "narco");
        Service.ajouterClient(c1);
        Service.ajouterClient(c2);
        Service.ajouterClient(c3);
        System.out.println("Verifier qu'il n'y a que 2 clients sur 3");
    }
    
    public static void testChercherPersonne() throws IOException, ParseException{
        
        System.out.println("début de testChercherPersonne");
        int id = Saisie.lireInteger("Entrez l'id :");
        String psw = Saisie.lireChaine("Entrez le mot de passe :");
        Personne p1 = Service.chercherPersonne(id, psw);
        if (p1!=null){
            System.out.println ("Bonjour " + p1.getNom());
        }
        else{
            System.out.println ("Identifiant ou mot de passe invalide !");
        }  
    }
    
    public static void testDemanderVoyance(){
        System.out.println("début de testDemanderVoyance");
        Personne cl1 = Service.chercherPersonne(11, "narco");
        Client c1 = Service.chercherClient(cl1);
        System.out.println(c1.getNom());
        
        Personne cl2 = Service.chercherPersonne(12, "narco");
        Client c2 = Service.chercherClient(cl2);
        System.out.println(c2.getNom());
        
        Medium m = Service.chercherMedium(1);
        System.out.println(m.getNom());
        
        System.out.println ("Tentative de voyance avec client 11 et medium 1");
        Voyance v1 = Service.demanderVoyance(m, c1);
        if (v1!=null){
             System.out.println ("Medium : "+v1.getMedium().getNom()+" Employe : "+v1.getEmploye().getNom()+" Client : "+v1.getClient().getNom());
        }else{
            System.out.println ("La voyance n'a pas pu être crée");
        }
        System.out.println ("Tentative de voyance avec client 12 et medium 1");
        Voyance v2 = Service.demanderVoyance(m, c2);
        if (v2!=null){
            System.out.println ("Medium : "+v2.getMedium().getNom()+" Employe : "+v2.getEmploye().getNom()+" Client : "+v2.getClient().getNom());
        }else{
             System.out.println ("La voyance n'a pas pu être créée");
        }
        System.out.println ("Tentative de voyance avec client 12 et medium 1");
        Voyance v3 = Service.demanderVoyance(m, c2);
        if (v3!=null){
            System.out.println ("Medium : "+v3.getMedium().getNom()+" Employe : "+v3.getEmploye().getNom()+" Client : "+v3.getClient().getNom());
        }else{
             System.out.println ("La voyance n'a pas pu être créée");
        }

    }
    
    public static void testChercherVoyance (){
        System.out.println("début de testChercherVoyance");
        System.out.println("Recherche de la voyance avec l'employé Jean Dupont, ayant l'Id 6");
        Personne emp = Service.chercherPersonne(6, "jdupont");
        Employe e = Service.chercherEmploye(emp);
        Voyance v = Service.chercherVoyance(e);
        if (v!= null)
            System.out.println ("Medium : "+v.getMedium().getNom()+" Employe : "+v.getEmploye().getNom()+" Client : "+v.getClient().getNom());
        else{
            System.out.println ("pas de voyance avec cet employé");
        }
    }
   
    public static void testDemarrerVoyance(){
        System.out.println("début de testDemarrerVoyance");
        Personne emp = Service.chercherPersonne(6, "jdupont");
        Employe e = Service.chercherEmploye(emp);
        Voyance v = Service.chercherVoyance(e);
        Service.demarrerVoyance (v);
   }
   
    public static void testDonnerPrediction() throws IOException{
       System.out.println("début de testDonnerPredictions");
        Personne cl = Service.chercherPersonne(11, "narco");
        Client c = Service.chercherClient(cl);
        if (c!= null){
            List<String> liste = Service.donnerPrediction (c, 3, 4, 2);
            for (String element : liste){
                System.out.println(element);
            }
        }
        else{
            System.out.println("Pas de client. Veuillez en ajouter un");
        }
   }
   
    public static void testEndVoyance() throws ParseException{
        System.out.println("début de testEndVoyance");
        SimpleDateFormat JSON_DATE_FORMAT = new SimpleDateFormat("'le' yyyy-MM-dd 'à' hh:mm:ss");
        Personne emp = Service.chercherPersonne(6, "jdupont");
        Employe e = Service.chercherEmploye(emp);
        Voyance v = Service.chercherVoyance(e);
        
        Service.endVoyance(v, JSON_DATE_FORMAT.parse("le 2018-12-02 à 21:13:12"), JSON_DATE_FORMAT.parse("le 2018-12-02 à 21:18:12"), "Un peu sceptique");
        System.out.println("Voyance enregistrée. Aller vérifier dans la base");
   }
   
    public static void testGetListEmploye(){
        System.out.println("début de testGetListEmploye");
        List<Employe> liste = Service.getListEmploye();
        for (Employe element : liste){
            System.out.println(element.getNom());
        }
   }
   
    public static void testGetListMedium(){
        System.out.println("début de testGetListMedium");
        List<Medium> liste = Service.getListMedium();
        for (Medium element : liste){
            System.out.println(element.getNom());
        }
   }

}
