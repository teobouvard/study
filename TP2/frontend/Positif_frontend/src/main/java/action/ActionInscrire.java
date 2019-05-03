/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package action;

import javax.servlet.http.HttpServletRequest;

/**
 *
 * @author tbouvard
 */
public class ActionInscrire extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {
        Integer id = Integer.parseInt(request.getParameter("login"));
        String password = request.getParameter("password");
        String civilite =  request.getParameter("civilite");
        String nom = request.getParameter("nom");
        String prenom = request.getParameter("prenom");
        String dateNaissance = request.getParameter("dateNaissance");
        String telephone = request.getParameter("telephone");
        String adresse = request.getParameter("poste");
        String mail = request.getParameter("mail");
        
        return true;
    }
    
}
