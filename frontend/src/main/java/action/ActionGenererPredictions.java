package action;

import java.io.IOException;
import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;
import javax.servlet.http.HttpServletRequest;
import javax.servlet.http.HttpSession;
import metier.data.Client;
import metier.data.Employe;
import metier.data.Voyance;
import metier.service.Service;
import static metier.service.Service.chercherVoyance;
import static metier.service.Service.donnerPrediction;

public class ActionGenererPredictions extends Action {

    @Override
    public boolean executer(HttpServletRequest request) {
        HttpSession session = request.getSession();
        Voyance voyance = chercherVoyance((Employe) session.getAttribute("personne"));
        Client client = voyance.getClient();
        Integer noteAmour = Integer.parseInt(request.getParameter("noteAmour"));
        Integer noteSante = Integer.parseInt(request.getParameter("noteSante"));
        Integer noteTravail = Integer.parseInt(request.getParameter("noteTravail"));

        List<String> listePredictions = null;
        try {
            listePredictions = donnerPrediction(client, noteAmour ,noteSante, noteTravail );
        } catch (IOException ex) {
            Logger.getLogger(ActionGenererPredictions.class.getName()).log(Level.SEVERE, null, ex);
        }
        
        request.setAttribute("liste-predictions", listePredictions);

        return true;
    }
}
