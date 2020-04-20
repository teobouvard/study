package action;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Employe;
import metier.data.Voyance;
import static metier.service.Service.chercherVoyance;
import static metier.service.Service.endVoyance;

public class ActionValiderVoyance extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {

        HttpSession session = request.getSession();
        Voyance voyance = chercherVoyance((Employe) session.getAttribute("personne"));
        String commentaire = (String) request.getParameter("commentaire");

        Date dateDebutAppel = null, dateFinAppel = null;
        SimpleDateFormat dateParser = new SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss");

        try {
            dateDebutAppel = dateParser.parse((String) request.getParameter("dateDebutAppel"));
            dateFinAppel = dateParser.parse((String) request.getParameter("dateFinAppel"));
        } catch (ParseException ex) {
            System.out.println("Erreur lors du parsing des dates");
            request.setAttribute("statut", Boolean.FALSE);
            return false;
        }

        if (endVoyance(voyance, dateDebutAppel, dateFinAppel, commentaire)) {
            request.setAttribute("statut", Boolean.TRUE);
        } else {
            request.setAttribute("statut", Boolean.FALSE);
        }
        return true;
    }
}
