package action;

import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Employe;
import metier.data.Voyance;
import static metier.service.Service.chercherVoyance;
import static metier.service.Service.demarrerVoyance;

public class ActionDemarrerVoyance extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {
        HttpSession session = request.getSession();
        Voyance voyance = chercherVoyance((Employe) session.getAttribute("personne"));
        demarrerVoyance(voyance);
        return true;
    }

}
