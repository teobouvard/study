/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package dao;

import javax.persistence.EntityManager;
import metier.data.Client;
import metier.data.Personne;

/**
 *
 * @author ghermet
 */
public class ClientDAO {
    
    public static void AjouterClient ( Client c ){
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.persist (c); 
    }
     public static void modifierClient(Client c) {
        EntityManager em = JpaUtil.obtenirEntityManager();
        em.merge(c);
    }
    // pas encore testé
    public static Client chercherClient (Personne unClient){
        EntityManager em = JpaUtil.obtenirEntityManager();
        Client c = em.find (Client.class,unClient.getId());
        return c;
    }
            
}
