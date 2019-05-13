package action;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Employe;
import metier.data.Voyance;
import metier.service.Service;

public class ActionChercherVoyance extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {
        HttpSession session = request.getSession();
        Voyance voyance = Service.chercherVoyance((Employe) session.getAttribute("personne"));
        if (voyance != null) {
            request.setAttribute("statut", Boolean.TRUE);
            request.setAttribute("voyance", voyance);
        } else {
            request.setAttribute("statut", Boolean.FALSE);
        }
        return true;
    }

}
