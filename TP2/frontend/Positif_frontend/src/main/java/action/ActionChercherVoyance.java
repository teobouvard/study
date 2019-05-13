/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package action;

import action.Action;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Employe;
import metier.data.Voyance;
import metier.service.Service;

/**
 *
 * @author tbouvard
 */
public class ActionChercherVoyance extends Action {


    @Override
    public boolean executer(HttpServletRequest request) {
       HttpSession session = request.getSession();
       Voyance voyance = Service.chercherVoyance((Employe)session.getAttribute("personne"));
       request.setAttribute("voyance",voyance);
       return true;
    }
    
}
